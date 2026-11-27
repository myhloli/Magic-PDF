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
