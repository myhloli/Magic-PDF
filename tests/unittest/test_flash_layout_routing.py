"""验证 Flash/TXT 页清单的批处理、成品替换与异常隔离。"""

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
import pytest

from docvortex.analyzers.native import PdfModel
from docvortex.document.pdf import PDFDocument
from mineru.backend.analysis.pdf import flash_layout, window
from mineru.config import get_config
from reportlab.pdfgen.canvas import Canvas


def test_layout_initialization_failure_keeps_native_and_skips_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """无权重或设备初始化失败时，原生结果及尚未分配的页图均保留原状态。"""
    monkeypatch.setattr(flash_layout, "_load_layout_model", MagicMock(side_effect=RuntimeError("missing weights")))
    render = MagicMock()
    monkeypatch.setattr(flash_layout, "load_images_from_pdf_bytes_range", render)
    pages = [[{"type": "text", "content": "native text"}]]
    assert flash_layout.replace_flash_layout_pages(MagicMock(), pages, [0], 1) == set()
    assert pages == [[{"type": "text", "content": "native text"}]]
    render.assert_not_called()


@pytest.mark.parametrize(
    ("threshold", "selected", "failure"),
    [
        (0.80, [0, 1, 3], False),
        (0.70, [1, 3], False),
        (0.70, [1, 3], True),
        (0.43, [3], False),
        (0.4299, [], False),
        (0.0, [], False),
        (1.0, [0, 1, 3], False),
    ],
)
def test_sparse_layout_uses_real_page_and_crops_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: float, selected: list[int], failure: bool
) -> None:
    """用同一组逐页分数验证阈值边界、满分页保护，以及推理失败页的原生回退。"""
    path = tmp_path / "input.pdf"
    canvas = Canvas(str(path), pagesize=(200, 200))
    for i in range(4):
        canvas.drawString(10, 160, f"Native page {i}")
        canvas.showPage()
    canvas.save()
    monkeypatch.setattr(get_config().flash, "layout_confidence_threshold", threshold)

    def predict(model: PdfModel, document: PDFDocument, *, collect_layout_diagnostics: bool) -> list[list[dict]]:
        """模拟原生模型按页序输出置信度和容易识别的文本。"""
        assert collect_layout_diagnostics and len(document) == 4
        model.layout_page_confidences = (0.80, 0.70, 1.0, 0.43)
        return [[{"type": "text", "content": f"baseline-{i}"}] for i in range(4)]

    def render(data: bytes, *, document: PDFDocument, start_page_id: int, end_page_id: int) -> list[dict]:
        """保留离散物理页的身份，以颜色复核模型及裁图接收的页号。"""
        assert start_page_id == end_page_id
        rendered.append(start_page_id)
        return [{"img_pil": Image.new("RGB", (400, 400), (start_page_id, 0, 0))}]

    def infer(images: list[Image.Image], batch_size: int) -> list[list[dict]]:
        """返回整页预测；异常模式下只有物理索引 3 的图片无法推理。"""
        ids = [im.getpixel((0, 0))[0] for im in images]
        calls.append(ids)
        assert batch_size == len(ids)
        if failure and 3 in ids:
            raise RuntimeError("single bad page")
        return [[{"label": "text", "bbox": [0, 0, 400, 170]}, {"label": "image", "bbox": [0, 200, 400, 400]}] for _ in ids]

    rendered, calls = [], []
    monkeypatch.setattr(PdfModel, "layout_page_confidences", (), raising=False)
    monkeypatch.setattr(PdfModel, "predict", predict)
    monkeypatch.setattr(flash_layout, "load_images_from_pdf_bytes_range", render)
    load_model = MagicMock(return_value=("cpu", MagicMock(batch_predict=infer)))
    monkeypatch.setattr(flash_layout, "_load_layout_model", load_model)
    # 成功页的实际 JPEG 已就绪，最终补图服务只需检查剩余原生块。
    final_crops = MagicMock()
    monkeypatch.setattr(window, "attach_visual_block_images_from_pdf", final_crops)
    with PDFDocument(path.read_bytes()) as document:
        pages = window.process_pdf_windows(
            document.bytes,
            document,
            effort="flash",
            parse_mode="txt",
            image_analysis=False,
            flash_txt_mode=True,
            hybrid_model=None,
            vlm_predictor=None,
        )
    assert load_model.call_count == int(bool(selected))
    assert rendered == selected
    if failure:
        assert calls == [[1, 3], [1], [3]]
    else:
        assert [index for batch in calls for index in batch] == selected
    accepted = set(selected)
    if failure:
        accepted.remove(3)
    for i, blocks in enumerate(pages):
        if i in accepted:
            assert f"Native page {i}" in str(blocks[0]["content"])
            assert blocks[1]["image_base64"].startswith("data:image/jpeg;base64,")
            assert final_crops.call_args.args[1][i] == []
        else:
            assert blocks == [{"type": "text", "content": f"baseline-{i}"}]


def test_routing_logs_structured_domain_summary_and_handles_legacy_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新版 docvortex 提供 layout_page_assessments 时输出分域路由日志，旧版模型仅凭置信度仍可正常回落。"""
    from types import SimpleNamespace

    path = tmp_path / "routing_logs.pdf"
    canvas = Canvas(str(path), pagesize=(200, 200))
    for i in range(3):
        canvas.drawString(10, 160, f"Log page {i}")
        canvas.showPage()
    canvas.save()
    monkeypatch.setattr(get_config().flash, "layout_confidence_threshold", 0.80)

    def run_once(*, with_assessments: bool, with_order_repairs: bool = False) -> tuple[list[str], list[str]]:
        info_logs: list[str] = []
        debug_logs: list[str] = []

        def predict(model: PdfModel, document: PDFDocument, *, collect_layout_diagnostics: bool) -> list[list[dict]]:
            assert collect_layout_diagnostics and len(document) == 3
            model.layout_page_confidences = (0.75, 1.0, 0.40)
            if with_assessments:
                model.layout_page_assessments = (
                    SimpleNamespace(primary_domain="spatial_collision", reasons=("spatial_collision: image_table×1(53%)",)),
                    SimpleNamespace(primary_domain=None, reasons=()),
                    SimpleNamespace(primary_domain="fragmentation", reasons=("fragmentation: 两个独立公式存在显著交叠",)),
                )
            else:
                model.layout_page_assessments = ()
            if with_order_repairs:
                model.extensions = {
                    "docvortex_pdf_structure": {
                        "order_repairs": [
                            {"page_idx": 0, "action": "figure_inversion"},
                            {"page_idx": 0, "action": "front_folio_relocate"},
                            {"page_idx": 2, "action": "severe_non_ua_vertical_fallback"},
                        ]
                    }
                }
            else:
                model.extensions = {}
            return [[{"type": "text", "content": f"native-{i}"}] for i in range(3)]

        monkeypatch.setattr(PdfModel, "layout_page_confidences", (), raising=False)
        monkeypatch.setattr(PdfModel, "layout_page_assessments", (), raising=False)
        monkeypatch.setattr(PdfModel, "predict", predict)
        monkeypatch.setattr(flash_layout, "replace_flash_layout_pages", MagicMock(return_value={0, 2}))
        monkeypatch.setattr(window, "attach_visual_block_images_from_pdf", MagicMock())
        monkeypatch.setattr(window.logger, "info", lambda msg, *args: info_logs.append(msg.format(*args)))
        monkeypatch.setattr(window.logger, "debug", lambda msg, *args: debug_logs.append(msg.format(*args)))
        with PDFDocument(path.read_bytes()) as document:
            window.process_pdf_windows(
                document.bytes,
                document,
                effort="flash",
                parse_mode="txt",
                image_analysis=False,
                flash_txt_mode=True,
                hybrid_model=None,
                vlm_predictor=None,
            )
        return info_logs, debug_logs

    info_new, debug_new = run_once(with_assessments=True, with_order_repairs=True)
    assert info_new == [
        (
            "Flash TXT applied tagged reading-order repairs on 2 pages "
            "(figure_inversion=1, front_folio_relocate=1, severe_non_ua_vertical_fallback=1)"
        ),
        "Flash TXT used PP layout on 2/2 flagged pages (fragmentation=1, spatial_collision=1)",
    ]
    assert len(debug_new) == 2
    assert "domain=spatial_collision" in debug_new[0] and "image_table" in debug_new[0]
    assert "domain=fragmentation" in debug_new[1] and "两个独立公式" in debug_new[1]

    info_legacy, debug_legacy = run_once(with_assessments=False)
    assert info_legacy == ["Flash TXT used PP layout on 2/2 flagged pages"]
    assert len(debug_legacy) == 2
    assert all("domain=" not in line for line in debug_legacy)


def test_pp_replacement_safety_guards_reject_wipeout_and_severe_reversal() -> None:
    """PP 漏检大面积原生正文或将边界页反向拆碎时拒绝替换，已由图/表承接的转换正常放行。"""
    native_page = [
        {"type": "text", "bbox": [0.1, 0.1, 0.9, 0.45], "content": "x" * 300},
        {"type": "text", "bbox": [0.1, 0.5, 0.9, 0.85], "content": "y" * 300},
    ]
    # 1. 空 PP 输出拦截
    assert "empty_pp_output" in (flash_layout._check_pp_replacement_safety(native_page, []) or "")
    # 2. A16 模式：PP 仅框出局部，大段原生正文落空无任何 PP 框覆盖 -> 拦截
    partial_pp = [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "header"}]
    assert "uncovered_text_loss" in (flash_layout._check_pp_replacement_safety(native_page, partial_pp) or "")
    # 3. 合理转换：PP 用 image/table 完整覆盖原文字区域 -> 放行（无误拒）
    covered_pp = [
        {"type": "image", "bbox": [0.08, 0.08, 0.92, 0.48], "content": ""},
        {"type": "table", "bbox": [0.08, 0.48, 0.92, 0.88], "content": "<table><tr><td>ok</td></tr></table>"},
    ]
    assert flash_layout._check_pp_replacement_safety(native_page, covered_pp) is None
    # 4. A20 模式：边界原生页(0.80)被 PP 反向碎裂成逐行 5 块正文(0.40) -> 拦截；原生本身已严重(0.40)时不拦截
    shattered_pp = [
        {
            "type": "text",
            "bbox": [0.12, y, 0.68, y + 0.018],
            "content": "a continuous line of the same passage",
            "lines": [{"bbox": [0.12, y, 0.68, y + 0.018]}],
        }
        for y in (0.12, 0.14, 0.16, 0.18, 0.20)
    ]
    borderline_native = [{"type": "text", "bbox": [0.12, 0.12, 0.68, 0.22], "content": "short"}]
    assert "severe_fragmentation_reversal" in (
        flash_layout._check_pp_replacement_safety(borderline_native, shattered_pp, native_confidence=0.80) or ""
    )
    assert flash_layout._check_pp_replacement_safety(borderline_native, shattered_pp, native_confidence=0.40) is None
    # 5. 嵌套 list 块仅统计真实可见字符，不得将内部字典 repr 计入字符数导致误拒
    nested_list_page = [
        {
            "type": "list",
            "bbox": [0.1, 0.1, 0.9, 0.5],
            "content": [
                {"type": "text", "content": [{"type": "text", "content": f"item {idx}"}]},
                {"type": "list", "content": [{"type": "text", "content": f"sub {idx}"}]},
            ],
        }
        for idx in range(5)
    ]
    assert flash_layout._visible_nonws_chars(nested_list_page[0]) == len("item0sub0")
    assert flash_layout._check_pp_replacement_safety(nested_list_page, []) is None


def test_replace_flash_layout_pages_keeps_native_when_guard_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """安全门禁拒绝替换时保留原生页、不加入 completed，且跳过该页的 PP 素材裁图。"""
    native_pages = [
        [{"type": "text", "bbox": [0.1, 0.1, 0.9, 0.8], "content": "a" * 400}],
    ]
    fake_model = MagicMock(batch_predict=MagicMock(return_value=[[{"label": "text", "bbox": [0, 0, 10, 10]}]]))
    monkeypatch.setattr(flash_layout, "_load_layout_model", MagicMock(return_value=("cpu", fake_model)))
    monkeypatch.setattr(
        flash_layout,
        "load_images_from_pdf_bytes_range",
        MagicMock(return_value=[{"img_pil": Image.new("RGB", (200, 200))}]),
    )
    monkeypatch.setattr(
        "mineru.backend.analysis.pdf.flash_layout_fill.fill_flash_layout_page",
        MagicMock(return_value=[{"type": "text", "bbox": [0.0, 0.0, 0.05, 0.05], "content": "tiny"}]),
    )
    attach_mock = MagicMock()
    monkeypatch.setattr(flash_layout, "attach_visual_block_images", attach_mock)
    doc = MagicMock()
    doc.bytes = b"%PDF-1.4"
    completed = flash_layout.replace_flash_layout_pages(doc, native_pages, [0], 1, native_confidences={0: 0.72})
    assert completed == set()
    assert native_pages[0][0]["content"] == "a" * 400
    attach_mock.assert_not_called()
