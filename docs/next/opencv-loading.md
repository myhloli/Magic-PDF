# OpenCV 按需加载与使用验收

2026-10-08。第三轮已完全移除 MinerU 自身的 OpenCV 导入和调用。下方第二轮及第一轮章节保留历史验收记录。

## 第三轮：调用代码完全移除

生产目录 `mineru/` 中已无 `cv2` 标识、导入或调用；基础依赖不包含 OpenCV，Torch/full 等 extra
保持原样，其他依赖自行引入 OpenCV 的行为仍不在清理范围。DocVortex 公共 API 和内核无需继续修改。

正常印章裁剪一直显式使用 `mode="homography"`。本轮删除没有产品调用方的相机标定支路：
`PlanB`、虚拟相机、畸变投影、`Rodrigues`、`remap`、两处 `calibrateCamera` 以及损失阈值重试。
`CurveTextRectifier`、`AutoRectifier` 默认改为 homography，保留原分段展开、短多边形外接框裁图、
几何异常回退和诊断绘图。显式请求 calibration 模式会抛出 `ValueError`，不静默更换算法；
内部 `loss_thresh` 参数不再提供。公共 Parser、产品参数和九种输出契约保持原样。

同时移除未使用的 `base64_to_cv2` 内部辅助函数、`imresize` 的旧 `backend="cv2"` 别名，
修正三处过时模块注释。旧标定白名单改为全生产目录禁止 OpenCV 的静态守卫。

测试对照在删除前冻结为 `tests/fixtures/model_image_reference.json`：29 个模型数组来自原 OpenCV
对照，15 组印章裁图及诊断图来自原显式 homography 路径。记录基准提交 `e6d89a6a`、原测试
源码摘要及参考库版本。测试代码也不再导入或调用 OpenCV，只保留导入阻断、静态审计和元数据检查。

| 验收 | 结果 |
| --- | --- |
| 静态生产及测试调用审计 | OpenCV 导入和直接调用均为 0；生产源码 `cv2` 文本匹配为 0 |
| 针对数值与跨库边界 | 77 项通过；包括 Python/Rust 两后端的横竖曲线、四点、短多边形和退化回退 |
| 整组回归阻断 OpenCV | 同一 77 项通过，冻结参考测试无需安装 OpenCV |
| 普通宿主回归 | 3043 passed、4 skipped；范围为 `not remote and not full_stack` |
| 实际 Flash OCR | demo1 第 5–6 页，候选阻断 cv2；完整 ModelJson、MiddleJson、九种输出和实际模型输入摘要与第二轮候选相同 |
| Ruff / 补丁 | 核心印章模块与修改后边界测试 E/F/W/ANN、格式检查通过；`git diff --check` 通过 |

宿主回归有 63 条警告，包含此前出现过的 Gradio 页范围测试事件循环析构告警。
该文件按 `PytestUnraisableExceptionWarning` 作为错误重跑，59 passed、3 skipped，未复现；
原始告警和重跑日志均保留，不过滤它。

原始证据位于 `/tmp/mineru-opencv-zero-20261008`，归档为
[output/opencv-zero-20261008/evidence.tar.gz](../../output/opencv-zero-20261008/evidence.tar.gz)。
本轮修改提交并合并到本地 dev，未推送或发布。第二轮归档保持原样。

## 第二轮：基础依赖移除

`project.dependencies` 不再包含任何 OpenCV 包。基础 ONNX 和 llama.cpp 推理、Torch/full extras、
以及 macOS Apple Silicon 自动选择 Torch extra 的声明保持原样；这些 extra 或其他依赖自行引入
OpenCV 的安装行为不在本轮清理范围。

MinerU 的常规 Torch/ONNX 预处理、OCR、公式、表格和印章 Homography 已通过
`docvortex.image` 公共接口使用共享数值内核。通用算法属于 DocVortex，模型阈值、排序和规则
继续属于 MinerU。解码、缩放、灰度、透视/仿射、轮廓、最小矩形、连通域、形态学、
多边形和推理抗锯齿线均不根据 cv2 是否安装更换算法。诊断图像使用 Pillow。

剩余四处直接调用仅限 `CurveTextRectifier.spatial_transform` 与 `calibrate` 两个旧相机标定方法，
使用函数内按需导入；它们不在产品印章 Homography 的实际调用链中。静态测试白名单限定到方法，
不放宽整个模块。公共 Parser、档位、参数、schema 2.0、旧持久缓存边界及九种渲染契约保持原样。

本轮需要新增公共图像接口，MinerU 的 DocVortex 最低版本提高为 `>=0.5.12,<1`。配套源码位于
`/Users/myhloli/.codex-workspaces/worktrees/docvortex-opencv-free-20261008/docvortex`；这是从已发布
0.5.11 建立的独立工作树，实现和验收期间原 main 未修改。验收后的双仓库改动现已分别
提交并合并到本地 MinerU dev、DocVortex main；0.5.12 和私有原生协议 32 是本地候选，尚未发布。
直接从索引安装此 checkout 前，应先发布或本地安装配套 DocVortex；当前指定环境已安装配套轮子。

| 验收 | 结果 |
| --- | --- |
| MinerU 完整测试 | 3010 passed、4 skipped；62 条既有第三方警告 |
| DocVortex 完整 Python 测试 | 9642 passed、14 skipped；6 条既有警告 |
| Rust workspace | 16 项通过；Clippy 检查通过 |
| 新公共图像接口 | 65 组独立 OpenCV 冻结数值样本，两后端验证；包含宽图分块舍入和边界粗线 |
| 真实 Torch、ONNX | 模型输入张量摘要、中间协议和九种输出相同；候选阻断 cv2 |
| 真实表格、Flash OCR、GGUF standard/advanced | 相同的实际产物和已采集输入摘要；候选阻断 cv2 |
| Python 参考后端 + 实际 ONNX | 输入张量、中间协议和九种输出相同；未加载 cv2 |
| Wheel 元数据 | MinerU 无 OpenCV 直接声明，ONNX/llama.cpp 仍在基础依赖 |
| 视觉复核 | 源 PDF 及前后导出页 5–6 已查看，无新增视觉差异；既有竖排表格导出版式局限未改动 |

数值校准专门覆盖 float32 灰度的向量/尾部舍入、三次采样计算精度、宽图通道块步长，以及
小型矩阵主元消元的融合乘加。这些差分在最终模型输入验证前完成修正。普通测试不要求 OpenCV：
新数值参考冻结为 fixture，历史运行时 OpenCV 对照通过 `importorskip` 运行。

原始证据为 `/tmp/mineru-opencv-free-20261008`：`identity.json`、`mineru-baseline/` 和
`baseline-site/` 冻结第一轮完成状态及 DocVortex 0.5.11；`acceptance.json` 保存七条验收比较；
`real-*/` 保存协议、九种输出、实际输入和后端信息；`production-opencv-audit.json` 保存白名单审计；
`wheel-metadata.json` 保存构建轮子依赖；`performance-summary.json` 与 `performance/` 保存成对计时。
`mineru-complete.patch`、`docvortex-complete.patch` 包含新增文件，配套轮子位于 `wheels/`。

每条路径使用五对交替运行的独立进程，首份后重新解析三次作为稳态样本。
计时包含实际解析，九种渲染与输入摘要在计时外；不复用解析结果，模型缓存按产品正常方式复用。
默认 Rust 后端的实际扩展路径与摘要写入记录，候选进程阻断 cv2。

| 路径（稳态中位数） | 第一轮完成状态 + DocVortex 0.5.11 | 本轮候选 | 变化 |
| --- | ---: | ---: | ---: |
| Flash txt | 0.2210 s | 0.2203 s | -0.31% |
| 基础 ONNX OCR | 3.8631 s | 4.0019 s | +3.59% |
| 基础 Torch OCR | 3.2700 s | 3.3559 s | +2.62% |
| ONNX 表格页 | 5.3238 s | 5.4964 s | +3.24% |

四条路径满足 <=5% 的稳态回归门槛，所有配对协议和九种产物相同。第一次表格轮次为 +5.13%，
因此继续优化大型连通区域：float32 坐标以连续字节传入 Rust，避免逐点创建 Python float/list；
凸包排序仍按完整坐标和原始索引裁决。241 项相关数值及私有接口测试、Clippy 通过后，
重测五对表格得到 +3.24%。失败轮次与修正后原始数据均保留，不混合两轮样本。

首次解析中位数变化分别为 Flash -0.50%、ONNX +0.36%、Torch +0.61%、表格 +1.31%。
这些结果仅针对本机样本和环境，不修改 README 的性能承诺；远端跨平台 CI 尚未执行。

原始证据归档到 [output/opencv-free-20261008/evidence.tar.gz](../../output/opencv-free-20261008/evidence.tar.gz)，
完整双仓库补丁和本地配套轮子一并保存，SHA-256 位于同目录 evidence.sha256。
归档保留分支整合前的验收快照。当前源码已提交并合并到本地 dev/main，未推送或发布。


## 第一轮记录

## 实现边界

Flash txt / auto→txt 在分类后直接进入 DocVortex 原生解析与按需补图，PDF 编排不再提前导入 Hybrid、OCR、表格和公式阶段。模型类型只在 TYPE_CHECKING 中导入，模型工厂仅导入当前需要的实现。混合用途模块中的 OpenCV 内核通过函数内普通导入加载；示例读图与调试绘图同样只在执行时导入。

后端文本回填、OCR、公式和表格，以及模型输入适配中的 RGB/BGR 交换改为 NumPy。三/四通道转换忽略 alpha，灰度输入复制为三通道，支持 uint8、uint16、float32，输出保持独立连续数组。模型侧仅新增 `model.ocr.image.rgb_to_bgr` 辅助函数；公共 Parser、分析接口、产品参数和文档协议保持原样。

方向评分的直角旋转、split/merge、既有 alpha 浮点合成、公式掩码裁边和 UniMERNet 零填充使用 NumPy。UNet 仍原位交换通道，保留随后 OpenCV 浮点减乘的原计算顺序。单通道 HWC 裁边和缩放后维度有专门回归断言。

保留的内核包括：图像解码、整数/浮点灰度转换、LINEAR/CUBIC 等缩放、透视/仿射变换、轮廓/连通域/形态学、印章矫正、表格 alpha 路径中的位运算与饱和加法，以及调试绘图。这些操作没有改动插值、边界、阈值或模型配置。high/xhigh 即使使用远程 VLM，仍可能因本地小模型加载 OpenCV。

静态 AST 审计的直接 `cv2.*()` 调用由 152 处降至 112 处，涉及文件由 26 个降至 19 个；顶层 `import cv2` 由 26 个降至 0。计数包含可选模型和调试代码，不代表单次解析调用次数。

## 验证

新增 `tests/unittest/test_opencv_boundaries.py`，现有 `docvortex-boundary` CI 矩阵纳入该文件。测试以独立进程阻断 cv2，避免测试顺序或已加载模块掩盖泄漏。

- 实际 Flash txt / auto→txt，同步与异步入口、非连续选页、含图补图、九种渲染和结果包保存均未加载 cv2，也未加载 Hybrid 模型管理器。
- 109 项现有非 PDF 原生回归在阻断 cv2 的进程中通过，包括 Office、EPUB、HTML、OFD、CSV 和 RTF 等现有格式路径。
- 31 项新增边界和像素测试通过：通道/位深/非连续数组、alpha 截断、旋转与所有权、透视内核、公式灰度与裁边、表格分类张量、UNet 输入和 UniMERNet 单通道填充。Torch 专属处理器测试在无 Torch 的 CI 环境跳过。
- 最终普通宿主回归 **3007 passed、4 skipped**，修改前基线独立复跑为 2976 passed、4 skipped；执行范围为 `tests/unittest -m 'not remote and not full_stack'`。真实本地模型另行对照，不替换推理结果。
- 最后完整回归含两条 Gradio 页范围测试期间的 BaseEventLoop 析构告警；该文件以 PytestUnraisableExceptionWarning 作为错误复跑，59 passed、3 skipped，未复现。记录保留，不修改或过滤该 Gradio 告警。
- 新增测试和验收脚本通过完整 Ruff lint/format；已修改旧文件相对基线没有新增 Ruff finding，旧文件已有 488 项 lint finding 保持原样。`git diff --check` 通过。

真实模型对照固定同一环境、模型文件和输入，对照两侧的完整 ModelJson、MiddleJson、九种输出与采集到的模型调用输入摘要：

| 路径 | 实际输入与后端 | 结果 |
| --- | --- | --- |
| basic / txt | demo2 第 1–2 页；ONNX 小模型 | 四类比较全部一致 |
| basic / txt | demo2 第 1–2 页；Torch CPU 小模型 | 四类比较全部一致 |
| standard / txt | demo2 第 1–2 页；ONNX + 本地 GGUF VLM | 四类比较全部一致 |
| advanced / txt | demo2 第 1–2 页；ONNX + 本地 GGUF 两阶段 VLM | 四类比较全部一致 |
| basic / txt 表格 | demo1 第 5–6 页；实际分类、SLANet、UNet | 四类比较全部一致 |
| Flash / ocr | demo1 第 5–6 页；方向 det/rec、OCR det/rec、表格投影 | 四类比较全部一致 |

ModelJson/MiddleJson 按完整 JSON 比较，包含内嵌素材；文本和结构输出按内容摘要比较。DOCX/EPUB 比较每个 ZIP 成员的解压内容，只归一化核心元数据中明确的创建时间。PDF 比较每页实际渲染像素，避免 PDF 时间字段影响结论。代表性正文/公式/彩色图和旋转表格源页与导出页已视觉检查；候选与基线相同，不据此扩大已有 OCR 精度承诺。

## 性能

macOS / arm64，Python 3.14.4，DocVortex 0.5.11，OpenCV 5.0.0.93；两侧使用同一安装环境，仅隔离 MinerU 源码。五对独立进程，轮换先后顺序，每进程首先转换一份三页含图 PDF，然后再真正解析十次。每轮通过公开 PdfModel.predict 记录 11 次实际原生入口调用、33 页请求，全部协议、渲染和素材比较一致。基线加载 cv2，候选全程未加载。

| 指标，中位数 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 准备模块和公开 PDF 接口 | 0.1933 秒 | 0.1926 秒 | 约持平 |
| 首份三页含图 parse，含延迟导入与补图 | 0.4081 秒 | 0.3066 秒 | -24.87% |
| 十次完整解析，共 30 页 | 0.1825 秒 | 0.1884 秒 | +3.26% |

稳态比例 1.0326，满足 ≤1.05 门槛。计时不含九种 renderer 和文件摘要；首份 parser 返回的协议与素材在计时内。准备模块包括解释器启动后脚本初始化和公开接口加载，不等于解释器启动耗时。两者相加的同进程中位数为 0.6014 → 0.4992 秒（-17.00%）。先前使用引擎诊断计数的五对实验每轮实测新增 33 次原生文本快照，记录另存 performance-summary-native-counter.json；正式交付脚本只依赖公开接口。未清空 OS 缓存或使用新安装位置，因此本表不是冷盘/首次安装结果，也不沿用 DocVortex 旧实验的 78% 收益。此处只报告该 Flash 样本的性能，模型对照的墙钟不用于吞吐结论。

## 复跑与证据

使用指定环境 `/Users/myhloli/projects/20240809magic_pdf/Magic-PDF/.venv4`。本轮对齐 DocVortex 0.5.11、mineru-vl-utils 2.0.5，并补装 pytest-cov；未删除 OpenCV。验证脚本 `tests/benchmarks/opencv_paths.py` 支持隔离源码、txt/ocr、档位、Torch/ONNX、选页及重复次数，并记录源文件摘要、加载状态、原生调用/页请求数、环境和产物。

```sh
.venv4/bin/python -m pytest tests/unittest -o addopts='' -q -m 'not remote and not full_stack'
.venv4/bin/python -m pytest tests/unittest/test_opencv_boundaries.py -o addopts='' -q
.venv4/bin/python tests/benchmarks/opencv_paths.py \
  --root /path/to/checkout --out /tmp/opencv-check \
  --tier basic --small-backend onnx --mode txt \
  --source demo/pdfs/demo1.pdf --pages 5-6
```

原始证据位于本机 `/tmp/mineru-opencv-20261008`：`identity.json` / `baseline.tar` 冻结修改前源码；`acceptance.json` 汇总比较；`real-*/` 保存真实模型产物和输入摘要；`performance-summary.json` / `performance-pairs.log` / `performance/` 保存五对时间和输出；`audit.json` 保存直接调用审计；`pixel-tests-final2.log` / `nonpdf-blocked.log` / `full-delivery.log` / `baseline-full.log` / `gradio-warning-replay.log` 保存测试；`lint-difference.json` 保存 lint 基线差分。原始记录已归档到 [output/opencv-20261008/evidence.tar.gz](../../output/opencv-20261008/evidence.tar.gz)，SHA-256 位于同目录 evidence.sha256；归档含修改前源码 tar 和所有原始比较/计时记录。

远端跨平台 CI 尚未执行；此次没有提交、推送或发布。
