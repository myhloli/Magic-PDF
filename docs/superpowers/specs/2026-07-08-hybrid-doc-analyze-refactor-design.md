# Hybrid doc_analyze Refactor Design

## 背景

`mineru/backend/hybrid/hybrid_analyze.py` 的 `doc_analyze` / `aio_doc_analyze` 当前同时承载入口参数归一、VLM runtime 初始化、按窗口加载图片、三档 effort 抽取、OCR det/rec、MFR、title split、append 和 finalize。随着 `medium`、`high`、`extra_high` 的语义逐步收敛，继续在入口函数里增加条件会让 `parse_method` 与 `effort` 的职责混在一起，后续维护时容易重新引入错误路径。

本次重构目标是把 `doc_analyze` 改成清晰的编排层，保留当前 public contract 和输出行为，不新增公开参数，不改变下游 `PageInfo` / `middle_json` schema。

## 目标

- `doc_analyze` 的核心控制维度只保留 `effort` 和规范化后的 `parse_mode`。
- `parse_method="auto"` 在进入主流程前归一为 `parse_mode="ocr"` 或 `parse_mode="txt"`。
- 三档 effort 都在每个窗口先运行小模型 layout。
- 三档 effort 都使用 OCR 小模型 det 生成文本行 sidecar，并用小模型 layout 的行内公式 bbox 辅助切割。
- `medium` 完全不加载 VLM runtime。
- `high` 使用 `predictor.batch_extract_with_layout`。
- `extra_high` 使用 `predictor.batch_two_step_extract`。
- 保持同步 `doc_analyze` 与异步 `aio_doc_analyze` 的策略一致。

## 非目标

- 不拆分公开 API、CLI、Gradio、parser 层的参数 contract。
- 不恢复 `low` effort，也不增加兼容 alias。
- 不新增 formula/table 公开开关。
- 不重写 `HybridMagicModel`、`model_output_to_middle_json.py` 或最终渲染/export 逻辑。
- 不改变 VLM predictor 的接口。

## 术语

`parse_mode` 是 `parse_method` 归一后的内部值，只允许：

- `ocr`: 文档文本来自光学识别。
- `txt`: 文档文本来自 PDF 原生文本。

`use_vlm_text_content` 表示下游 `MagicModel` 是否信任 VLM 返回的文本内容。它不是“是否运行 OCR 小模型”的含义。

## 决策表

| effort | 主结构来源 | `parse_mode=ocr` 文本来源 | `parse_mode=txt` 文本来源 | 强制 VLM 解析块 | inline formula MFR | OCR det | OCR rec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `medium` | 小模型 layout | 小模型 OCR det + rec | PDF native text | 无 VLM | `ocr/txt` 都需要 | 需要 | 仅 `ocr` |
| `high` | 小模型 layout 约束 VLM | VLM block OCR | PDF native text + VLM 结构块 | `index` / `code` / 行间公式 / `table` | 仅 `txt` | 需要 | 不需要 |
| `extra_high` | VLM two-step 原生结果 | VLM block OCR | PDF native text + VLM 结构块 | `index` / `code` / 行间公式 / `table` | 仅 `txt` | 需要 | 不需要 |

由表可得：

- `use_vlm_text_content = parse_mode == "ocr" and effort in {"high", "extra_high"}`
- `needs_mfr = parse_mode == "txt" or effort == "medium"`
- `needs_ocr_rec = parse_mode == "ocr" and effort == "medium"`
- `parse_mode == "txt"` 不等于所有内容都走 PDF native text；只有可由 PDF native 回填的文本类块应被 VLM 跳过，强制 VLM 解析块必须继续交给 VLM。
- `high` 和 `extra_high` 共用同一个 `forced_vlm_block_types`，包含 `index` / `code` / 行间公式 / `table`；`extra_high` 的 two-step extract 通常不会单独产出 `index`，但统一集合可以避免两套策略漂移。

## 内部策略对象

新增内部 dataclass：

```python
@dataclass(frozen=True)
class _HybridAnalyzePlan:
    """记录 Hybrid 单次解析的规范化策略，避免 effort/parse_method 判断散落在流程中。"""

    effort: str
    parse_mode: Literal["ocr", "txt"]
    use_vlm_text_content: bool
    needs_mfr: bool
    needs_ocr_rec: bool
    extractor: Literal["medium_local", "high_with_layout", "extra_high_two_step"]
    forced_vlm_block_types: frozenset[str]
    vlm_not_extract_list: tuple[str, ...] | None
```

该对象只在 `hybrid_analyze.py` 内部使用，不作为公开 API。

## 阶段划分

### 1. 参数归一

`_resolve_parse_mode(pdf_doc, parse_method)`：

- `parse_method == "auto"` 时调用 `pdf_doc.classify()`。
- `parse_method == "ocr"` 时直接返回 `ocr`。
- `parse_method == "txt"` 时直接返回 `txt`。
- 非法值抛出 `ValueError`。

`_build_hybrid_analyze_plan(effort, parse_mode)`：

- 先调用 `validate_effort(effort)`。
- 按决策表返回 `_HybridAnalyzePlan`。
- `vlm_not_extract_list` 不能无条件等同于未来可能扩展的 `NOT_EXTRACT_TYPES`；它必须排除当前 plan 的 `forced_vlm_block_types`，避免误把 `index` / `code` / 行间公式 / `table` 交给 PDF native 或空内容回填。

### 2. Runtime 初始化

`medium` 不调用 `_load_vlm_runtime()`，并强制 `predictor=None`。

`high` / `extra_high` 才按需加载 VLM runtime：

- 同步入口继续使用 `ModelSingleton().get_model(...)`。
- 异步入口继续使用 `_get_model_async(...)`。
- `predictor_execution_guard` / `aio_predictor_execution_guard` 保持原有边界。

### 3. 每窗口统一前置 layout

每个 `_ProcessingWindow` 都先加载图片并运行本地 layout，产出：

- `images_pil_list`
- `np_images`
- `images_layout_res`
- `local_context`

本地 context 的 `formula_enable` 应由 plan 决定：

- `plan.needs_mfr=True` 时需要加载 MFR。
- `plan.needs_mfr=False` 时不加载 MFR，避免 high/extra_high OCR 模式多占资源。

### 4. Effort 抽取

`medium_local`：

- 直接基于 `images_layout_res` 构造 medium model list。
- 保留 medium 的表格、印章 OCR、formula_number OCR-rec 等本地增强逻辑。
- 不调用 VLM。

`high_with_layout`：

- 使用 `images_layout_res` 构造 `blocks_list`。
- 调用 `predictor.batch_extract_with_layout(...)`。
- `parse_mode=ocr` 时不传 `not_extract_list`，由 VLM 完成 block OCR。
- `parse_mode=txt` 时只跳过可由 PDF native 回填的文本类块；`index`、`code`、行间公式、`table` 必须留给 VLM 解析。

`extra_high_two_step`：

- 调用 `predictor.batch_two_step_extract(...)`。
- `parse_mode=ocr` 时不传 `not_extract_list`，由 VLM 完成 block OCR。
- `parse_mode=txt` 时只跳过可由 PDF native 回填的文本类块；`index`、`code`、行间公式、`table` 必须留给 VLM 解析。

`image_analysis` 继续只在 `extra_high` 中生效，`medium` 和 `high` 强制关闭，避免低/中资源路径隐式触发更重的 VLM 图像分析。

### 5. Sidecar 增强

OCR det 三档都执行，用于文本行 sidecar 和后续段落合并提示。

MFR 只在 `plan.needs_mfr=True` 时执行：

- `medium + ocr/txt` 都执行。
- `high/extra_high + txt` 执行。
- `high/extra_high + ocr` 不执行，行内公式由 VLM block OCR 负责。

OCR rec 只在 `plan.needs_ocr_rec=True` 时执行：

- 仅 `medium + ocr` 执行。
- `high/extra_high + ocr` 不执行，因为 VLM 已提供 block 文本。
- 所有 `txt` 模式不执行 OCR rec。

`high/extra_high + ocr` 的 OCR det sidecar 只保留空文本行提示，不能覆盖 VLM content。

`high/extra_high + txt` 的 inline formula MFR 只为 PDF native 文本类块补行内公式；已经强制 VLM 解析的 `index`、`code`、行间公式、`table` 内容仍以 VLM 输出为准。

### 6. Append 与 finalize

`append_pages(...)` 继续调用 `blocks_to_page_info(...)`，但传入的 `_ocr_enable` 可以保持当前兼容形态，对下游语义更清楚的变量是 `use_vlm_text_content`。

`_finalize_hybrid_middle_json(...)` 保持现有行为：

- `client_side_output_generation=True` 时走 `apply_server_side_postprocess(...)`。
- 否则走 `finalize_middle_json(...)`。

重构不改变 `finalize_middle_json_from_preproc(...)` 的 merge、跨页表格、title leveling、split-title normalize 行为。

## 同步与异步一致性

同步和异步入口应共享以下纯策略函数：

- `_resolve_parse_mode`
- `_build_hybrid_analyze_plan`
- 计算 `batch_ratio` 的规则
- 每个 effort 的参数选择规则

异步入口可以继续用 `asyncio.to_thread(...)` 包装本地 CPU/GPU 小模型步骤，但不能维护一套独立的 effort/parse 判断。

## 测试策略

### Plan 单测

新增或扩展 focused tests，覆盖 3 effort x 2 parse_mode：

- `medium + ocr`: `use_vlm_text_content=False`, `needs_mfr=True`, `needs_ocr_rec=True`
- `medium + txt`: `use_vlm_text_content=False`, `needs_mfr=True`, `needs_ocr_rec=False`
- `high + ocr`: `use_vlm_text_content=True`, `needs_mfr=False`, `needs_ocr_rec=False`
- `high + txt`: `use_vlm_text_content=False`, `needs_mfr=True`, `needs_ocr_rec=False`
- `extra_high + ocr`: `use_vlm_text_content=True`, `needs_mfr=False`, `needs_ocr_rec=False`
- `extra_high + txt`: `use_vlm_text_content=False`, `needs_mfr=True`, `needs_ocr_rec=False`

### Runtime 路由单测

沿用 `tests/unittest/test_hybrid_high_extra_high_ocr_routing.py` 的 fake predictor 风格：

- `medium` 不加载 VLM runtime。
- `high + ocr` 调 `batch_extract_with_layout`，不跑 MFR/rec，`use_vlm_text_content=True`。
- `high + txt` 调 `batch_extract_with_layout`，传 `not_extract_list`，跑 MFR，不跑 rec，并断言 `index` / `code` / 行间公式 / `table` 不在跳过列表中。
- `extra_high + ocr` 调 `batch_two_step_extract`，不跑 MFR/rec，`use_vlm_text_content=True`。
- `extra_high + txt` 调 `batch_two_step_extract`，传 `not_extract_list`，跑 MFR，不跑 rec，并断言 `index` / `code` / 行间公式 / `table` 不在跳过列表中。
- `medium + ocr` 跑 OCR det + OCR rec + MFR。

### 回归测试

保留并更新以下测试族：

- `tests/unittest/test_hybrid_analyze_refactor_guards.py`
- `tests/unittest/test_hybrid_medium_ocr_text_fill.py`
- `tests/unittest/test_hybrid_medium_formula_model_loading.py`
- `tests/unittest/test_hybrid_medium_local_layout_memory.py`
- `tests/unittest/test_hybrid_medium_table_inline_objects.py`
- `tests/unittest/test_hybrid_high_extra_high_ocr_routing.py`

## 验证命令

实现后至少运行：

```bash
.venv1/bin/python -m pytest -o addopts='' \
  tests/unittest/test_hybrid_analyze_refactor_guards.py \
  tests/unittest/test_hybrid_medium_ocr_text_fill.py \
  tests/unittest/test_hybrid_medium_formula_model_loading.py \
  tests/unittest/test_hybrid_medium_local_layout_memory.py \
  tests/unittest/test_hybrid_medium_table_inline_objects.py \
  tests/unittest/test_hybrid_high_extra_high_ocr_routing.py -q

.venv1/bin/python -m py_compile mineru/backend/hybrid/hybrid_analyze.py
git diff --check
```

如果实现触碰 `model_output_to_middle_json.py` 或 `hybrid_magic_model.py`，还需要补跑相关 focused tests，并明确说明触碰原因。

## 风险与约束

- `use_vlm_text_content` 的语义必须保持为“文本内容是否来自 VLM”，不能重新被解释为 OCR 开关。
- `medium` 的低资源边界不能被破坏，任何 VLM import 都不能进入 medium 路径。
- `high/extra_high + ocr` 不应触发 MFR 或 OCR-rec。
- `high/extra_high + txt` 必须保留 inline formula MFR，否则 PDF native text 路径会丢失行内公式补偿。
- `high + txt` 必须强制 VLM 解析 `index`、`code`、行间公式和 `table`，不能因 `not_extract_list` 扩展而退化为 PDF native 回填。
- `extra_high + txt` 必须强制 VLM 解析 `index`、`code`、行间公式和 `table`，不能因 `not_extract_list` 扩展而退化为 PDF native 回填；`index` 在 two-step 输出中通常不存在，但必须和 `high` 使用同一强制集合。
- `INDEX -> TEXT` 的归一仍应留在 `MagicModel` 内容构造完成之后，不能提前转换。
- 新增函数和方法必须包含中文注释，方便 review。
