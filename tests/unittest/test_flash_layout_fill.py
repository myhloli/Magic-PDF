# Copyright (c) Opendatalab. All rights reserved.
"""验证按 PP 页级框回填原生内容，并保留 Flash 视觉输出。"""

from io import BytesIO
from typing import Any

import pytest
from docvortex.document.pdf import PDFDocument
from docvortex.postprocess.document import model_json_to_middle_json
from docvortex.render import render_html
from docvortex.schema import ModelJson
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from mineru.backend.analysis.pdf.flash_layout_fill import fill_flash_layout_page


def _layout_item(label: str, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    """构造与 PP 输出字段一致的视觉页面像素框。"""
    return {"label": label, "bbox": list(bbox)}


def test_refill_uses_pp_order_native_lines_links_and_tables() -> None:
    """同一视觉行的粗体链接及正文保持次序，真实表格与投影兜底同时保留。"""
    buffer = BytesIO()
    pdf = Canvas(buffer, pagesize=(260, 200))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(15, 160, "Alpha")
    pdf.linkURL("https://example.com", (15, 156, 53, 174), relative=0)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(70, 160, "Bravo")
    pdf.drawString(15, 140, "Second row")
    pdf.drawString(80, 115, "x = 2")
    pdf.drawString(142, 115, "(1)")
    pdf.drawString(170, 115, "Inside image")
    pdf.drawString(15, 25, "Figure 1. Native legend")
    for x in (15, 80, 145):
        pdf.line(x, 100, x, 40)
    for y in (40, 60, 80, 100):
        pdf.line(15, y, 145, y)
    for row, y in enumerate((86, 66, 46), start=1):
        for col, x in enumerate((20, 90), start=1):
            pdf.drawString(x, y, f"{row}{col}")
    pdf.drawString(165, 50, "Fallback only")
    pdf.save()

    items = [
        _layout_item("figure_title", (6, 163, 160, 188)),
        _layout_item("image", (160, 72, 258, 110)),
        _layout_item("text", (7, 20, 140, 68)),
        _layout_item("table", (14, 98, 148, 162)),
        _layout_item("table", (156, 132, 254, 161)),
        _layout_item("display_formula", (78, 70, 129, 91)),
        _layout_item("formula_number", (140, 70, 159, 91)),
    ]
    with PDFDocument(buffer.getvalue()) as document:
        blocks = fill_flash_layout_page(document[0], items, (260, 200))

    assert [block["type"] for block in blocks] == ["caption", "image", "text", "table", "table", "equation"]
    assert blocks[0]["content"] == [{"type": "text", "content": "Figure 1. Native legend"}]
    assert blocks[1]["content"] == ""
    assert blocks[1]["bbox"] == pytest.approx([160 / 260, 72 / 200, 258 / 260, 110 / 200])
    assert blocks[2]["content"] == [
        {
            "type": "hyperlink",
            "url": "https://example.com",
            "content": [{"type": "text", "content": "Alpha", "styles": ["bold"]}],
        },
        {"type": "text", "content": " Bravo Second row"},
    ]
    assert len(blocks[2]["lines"]) == 2
    assert "<tr><td>21</td><td>22</td></tr>" in blocks[3]["content"]
    assert blocks[4]["content"] == "Fallback only"
    assert blocks[5]["content"] == ""
    assert blocks[5]["bbox"] == pytest.approx([78 / 260, 70 / 200, 159 / 260, 91 / 200])


def test_refill_visual_page_rotation_and_absent_embedded_text() -> None:
    """PP 框使用旋转后的页坐标，裁图尺寸非整数倍及区域无可见原字时仍保持版面。"""
    buffer = BytesIO()
    pdf = Canvas(buffer, pagesize=(250, 200))
    pdf.setPageRotation(90)
    pdf.drawString(15, 160, "Rotated")
    hidden = pdf.beginText(75, 145)
    hidden.setTextRenderMode(3)
    hidden.textLine("Hidden OCR layer")
    pdf.drawText(hidden)
    pdf.save()

    with PDFDocument(buffer.getvalue()) as document:
        page = document[0]
        assert page.rotation == 90
        blocks = fill_flash_layout_page(
            page,
            [_layout_item("text", (300, 20, 350, 130)), _layout_item("text", (270, 140, 330, 391))],
            (502, 397),
        )

    assert blocks[0]["angle"] == 90
    assert blocks[0]["content"] == [{"type": "text", "content": "Rotated"}]
    assert blocks[0]["bbox"] == pytest.approx([300 / 502, 20 / 397, 350 / 502, 130 / 397])
    assert len(blocks[0]["lines"]) == 1
    assert blocks[1]["content"] == []
    assert blocks[1]["lines"] == []


def test_refill_splits_native_line_between_adjacent_pp_regions() -> None:
    """同一源行跨 PP 框时按字符切分，两块文字各自只出现一次。"""
    buffer = BytesIO()
    pdf = Canvas(buffer, pagesize=(260, 200))
    pdf.drawString(15, 160, "Alpha Bravo")
    pdf.save()

    with PDFDocument(buffer.getvalue()) as document:
        blocks = fill_flash_layout_page(
            document[0],
            [_layout_item("text", (45, 20, 98, 60)), _layout_item("paragraph_title", (10, 20, 45, 60))],
            (260, 200),
        )

    assert [block["content"] for block in blocks] == [
        [{"type": "text", "content": "Bravo"}],
        [{"type": "text", "content": "Alpha"}],
    ]
    assert blocks[0]["lines"][0]["bbox"][0] >= 45 / 260
    assert blocks[1]["lines"][0]["bbox"][2] <= 45 / 260


def test_refill_uses_form_clip_for_titles_prose_and_table_projection() -> None:
    """裁剪 Form 内外的字形落在同一 PP 框时，标题、正文和表格投影只回填实际显示的字。"""
    buffer = BytesIO()
    pdf = Canvas(buffer, pagesize=(240, 160))
    pdf.beginForm("Swatch", 0, 0, 90, 160)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(15, 130, "Dimensions")
    pdf.drawString(130, 130, "Back width")
    pdf.drawString(130, 90, "Old paragraph")
    pdf.drawString(130, 50, "40 deg")
    pdf.endForm()
    pdf.doForm("Swatch")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(15, 90, "Made locally")
    pdf.drawString(15, 50, "30 deg")
    pdf.save()

    items = [
        _layout_item("paragraph_title", (10, 15, 220, 38)),
        _layout_item("text", (10, 55, 220, 78)),
        _layout_item("table", (10, 95, 220, 118)),
    ]
    with PDFDocument(buffer.getvalue()) as document:
        blocks = fill_flash_layout_page(document[0], items, (240, 160))
    assert [block["content"] for block in blocks] == [
        [{"type": "text", "content": "Dimensions"}],
        [{"type": "text", "content": "Made locally"}],
        "30 deg",
    ]


def test_refill_mixed_font_titles_render_with_original_word_boundaries() -> None:
    """模型标题保留真实分词和层级，从 ModelJson 转成 HTML 后仍可复制。"""
    buffer = BytesIO()
    pdf = Canvas(buffer, pagesize=(260, 200))
    for left, right, y in (("Hyper", "text", 170), ("Alpha ", "Bravo", 130)):
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(15, y, left)
        pdf.setFont("Helvetica", 16)
        pdf.drawString(15 + stringWidth(left, "Helvetica-Bold", 16), y, right)
    pdf.save()

    with PDFDocument(buffer.getvalue()) as document:
        blocks = fill_flash_layout_page(
            document[0],
            [_layout_item("doc_title", (1, 1, 259, 50)), _layout_item("paragraph_title", (1, 51, 259, 100))],
            (260, 200),
        )

    model = ModelJson(
        pages=[blocks],
        page_index_map=[],
        metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
    )
    middle = model_json_to_middle_json(model)
    html = render_html(middle, standalone=False)
    assert "Hypertext</h1>" in html
    assert "Alpha Bravo</h2>" in html
