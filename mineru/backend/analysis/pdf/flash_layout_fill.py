# Copyright (c) Opendatalab. All rights reserved.
"""将 PP-DocLayoutV2 的页级版面填充为 Flash TXT model-list。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from docvortex.analyzers.pdf import (
    PDFTextLine,
    apply_text_evidence,
    join_tight_text,
    prepare_table_page,
    prepare_text_evidence,
    project_native_table_text,
    recover_table_region,
)
from docvortex.content.text import merge_text_line_contents
from docvortex.document.pdf import Char, PDFPage
from docvortex.geometry import bbox_center, convert_bbox, normalize_bbox, normalize_quarter_turn_angle, rotate_bbox
from loguru import logger

from ....types import RAW_FORMULA_NUMBER, BBox, BlockType
from .constants import CODE_CONTENT_BLOCK_TYPES, LINE_METADATA_BLOCK_TYPES, VLM_LAYOUT_LABEL_MAP
from .formulas import optimize_hybrid_formula_number_blocks
from .text.native import _replace_private_use_chars

_IMAGE_TYPES = {BlockType.IMAGE, BlockType.CHART}
_NON_TEXT_TYPES = {*_IMAGE_TYPES, BlockType.TABLE, BlockType.EQUATION}


@dataclass(slots=True)
class _LayoutRegion:
    """保存 PP 坐标映射与由此区域独占的可见原生行。"""

    block: dict[str, Any]
    bbox: BBox
    explicit_angle: bool
    lines: list[PDFTextLine]


def _layout_regions(
    items: list[dict[str, Any]], page_size: tuple[float, float], image_size: tuple[int, int]
) -> list[_LayoutRegion]:
    """按 PP 已排序的输出构建区域，像素框按两轴实际图幅转换到视觉页面。"""
    regions = []
    for item in items:
        label = str(item.get("label") or item.get("type") or "")
        block_type = VLM_LAYOUT_LABEL_MAP.get(label)
        if block_type is None:
            continue
        bbox = normalize_bbox(item.get("bbox"))
        unit = convert_bbox(bbox, source_space="pixel", target_space="unit", page_size=image_size, clip=True)
        point = convert_bbox(unit, source_space="unit", target_space="point", page_size=page_size)
        if point is None:
            continue
        block: dict[str, Any] = {
            "type": block_type,
            "bbox": list(unit),
            "content": "",
            "angle": normalize_quarter_turn_angle(item.get("angle")),
        }
        if label == "seal":
            block["sub_type"] = "seal"
        regions.append(_LayoutRegion(block, point, item.get("angle") is not None, []))
    return regions


def _char_bbox(char: Char) -> BBox | None:
    """以实际墨迹定位字符，缺少紧框的字形与生成空格使用原生布局框。"""
    return normalize_bbox(char.get("tight_bbox")) or normalize_bbox(char["bbox"].bbox)


def _union_boxes(boxes: list[BBox]) -> BBox:
    """返回当前同一行内全部成员的页面包络。"""
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _line_fragment(line: PDFTextLine, chars: list[Char], *, code: bool) -> PDFTextLine | None:
    """按当前区域拥有的成员恢复行片段，切分后用真实字框生成行框。"""
    if not chars:
        return None
    if code:
        text = "".join(char["char"] for char in chars)
    elif len(chars) == len(line.chars):
        text = line.text
    else:
        text = join_tight_text(chars)
    text = _replace_private_use_chars(text.replace("\x02", "-").replace("\r", "").replace("\n", ""))
    text = text.rstrip() if code else text.strip()
    if not text.strip():
        return None
    bbox = line.bbox
    if len(chars) != len(line.chars):
        boxes = [box for char in chars if char["char"].strip() and (box := _char_bbox(char)) is not None]
        if boxes:
            bbox = _union_boxes(boxes)
    return PDFTextLine(bbox, line.angle, text, tuple(chars), line.row_index)


def _assign_native_lines(lines: tuple[PDFTextLine, ...], regions: list[_LayoutRegion]) -> None:
    """按字形墨迹中心分配行成员，优先使用重叠区域中范围更具体的 PP 框。"""
    ordered = sorted(regions, key=lambda region: (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]))
    for line in lines:
        lx0, ly0, lx1, ly1 = line.bbox
        candidates = [
            region
            for region in ordered
            if region.bbox[0] < lx1 and lx0 < region.bbox[2] and region.bbox[1] < ly1 and ly0 < region.bbox[3]
        ]
        if not candidates:
            continue
        members: dict[int, list[Char]] = {}
        previous_index = None
        for char in line.chars:
            if char.get("text_render_mode") in (3, 7) or char.get("text_is_visible") is False:
                previous_index = None
                continue
            box = _char_bbox(char)
            if box is None:
                # PDFium 的零宽词界属于前一个位置可用的原字，去首尾空白时不会泄漏到邻框。
                if previous_index is not None and char["char"].isspace():
                    members[previous_index].append(char)
                continue
            cx, cy = bbox_center(box)
            previous_index = None
            for index, region in enumerate(candidates):
                x0, y0, x1, y1 = region.bbox
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    members.setdefault(index, []).append(char)
                    previous_index = index
                    break
        for index, chars in members.items():
            region = candidates[index]
            block_type = region.block["type"]
            if block_type in _IMAGE_TYPES:
                continue
            fragment = _line_fragment(line, chars, code=block_type in CODE_CONTENT_BLOCK_TYPES)
            if fragment is not None:
                region.lines.append(fragment)


def _join_row_fragments(fragments: list[PDFTextLine], page_size: tuple[float, float], angle: int, *, code: bool) -> str:
    """按原生词界拼接同一视觉行，保留源空格并去除合成的中日韩空格。"""
    parts = [fragments[0].text]
    for left, right in pairwise(fragments):
        left_end = next(index for index in range(len(left.chars) - 1, -1, -1) if not left.chars[index]["char"].isspace())
        right_start = next(index for index, char in enumerate(right.chars) if not char["char"].isspace())
        boundary = (*left.chars[left_end:], *right.chars[: right_start + 1])
        left_char, right_char = boundary[0], boundary[-1]
        text = "".join(char["char"] for char in boundary) if code else join_tight_text(boundary)
        separator = text[len(left_char["char"]) : -len(right_char["char"])]
        if len(boundary) == 2 and not separator and right_char["char_idx"] != left_char["char_idx"] + 1:
            left_box = rotate_bbox(tuple(left_char["bbox"].bbox), *page_size, angle)
            right_box = rotate_bbox(tuple(right_char["bbox"].bbox), *page_size, angle)
            glyph_width = min(left_box[2] - left_box[0], right_box[2] - right_box[0])
            if right_box[0] - left_box[2] > max(0.5, 0.25 * glyph_width):
                separator = " "
        parts.extend((separator if code else (" " if separator else ""), right.text))
    return "".join(parts)


def _fill_text_region(region: _LayoutRegion, page_size: tuple[float, float]) -> None:
    """在区域正向坐标中对原生行排序，写入后处理需要的文本和物理行框。"""
    block = region.block
    angle = block["angle"]
    block_type = block["type"]
    code = block_type in CODE_CONTENT_BLOCK_TYPES
    by_row: dict[tuple[int, int], list[PDFTextLine]] = {}
    for index, line in enumerate(region.lines):
        # 同一原生视觉行的多字体片段沿横向排列，字号差异不能让它们交换顺序或各算一行。
        key = (line.angle, line.row_index if line.row_index is not None else -1 - index)
        by_row.setdefault(key, []).append(line)
    rows = []
    for fragments in by_row.values():
        fragments.sort(key=lambda line: rotate_bbox(line.bbox, *page_size, angle)[0])
        bbox = _union_boxes([line.bbox for line in fragments])
        local_bbox = rotate_bbox(bbox, *page_size, angle)
        text = _join_row_fragments(fragments, page_size, angle, code=code)
        rows.append(((local_bbox[1], local_bbox[0]), bbox, text))
    rows.sort(key=lambda row: row[0])
    texts = [text for _, _, text in rows]
    if code or block_type == BlockType.INDEX:
        block["content"] = "\n".join(texts)
    else:
        block["content"] = merge_text_line_contents(texts)
    if block_type in LINE_METADATA_BLOCK_TYPES or block_type == RAW_FORMULA_NUMBER:
        block["lines"] = []
        for _, bbox, _ in rows:
            # 墨迹可跨框边缘，物理行几何以最终 PP 区域为上限。
            x0, y0, x1, y1 = region.bbox
            clipped = normalize_bbox((max(bbox[0], x0), max(bbox[1], y0), min(bbox[2], x1), min(bbox[3], y1)))
            unit = convert_bbox(clipped, source_space="point", target_space="unit", page_size=page_size)
            if unit is not None:
                block["lines"].append({"bbox": list(unit)})


def fill_flash_layout_page(
    page: PDFPage, layout_items: list[dict[str, Any]], image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    """按 PP 区域和顺序回填单页可复制原生文字、表格和行内语义。"""
    page_size = page.size
    regions = _layout_regions(layout_items, page_size, image_size)
    if not any(region.block["type"] not in _IMAGE_TYPES for region in regions):
        return [region.block for region in regions]

    # PP 框来自实际渲染页，正文和表格共用同一份裁剪、遮盖后的字符几何。
    snapshot = page.get_snapshot(visible_only=True)
    tables = [region for region in regions if region.block["type"] == BlockType.TABLE]
    evidence = prepare_text_evidence(
        page,
        snapshot=snapshot,
        supported_angles=(0.0, 90.0, 180.0, 270.0),
        table_regions=[region.bbox for region in tables],
        include_lines=True,
    )
    _assign_native_lines(evidence.lines, regions)
    for region in regions:
        if region.lines and not region.explicit_angle:
            # 原生行角度已经包含页面 /Rotate，不再重复旋转视觉几何。
            region.block["angle"] = Counter(line.angle for line in region.lines).most_common(1)[0][0]
        if region.block["type"] not in _NON_TEXT_TYPES:
            _fill_text_region(region, page_size)

    if tables:
        table_page = prepare_table_page(page, geometry=evidence.geometry, vector_geometry=evidence.vector_geometry)
        for region in tables:
            angle = region.block["angle"]
            try:
                table = recover_table_region(table_page, region.bbox, angle=angle)
            except Exception as exc:
                logger.warning(
                    "Flash layout native table recovery failed: "
                    f"page_idx={snapshot.page_index}, bbox={region.bbox}, error={exc}"
                )
                table = None
            region.block["content"] = (
                table.html
                if table is not None and table.html.strip()
                else project_native_table_text(table_page, region.bbox, angle=angle)
            )

    blocks = optimize_hybrid_formula_number_blocks([region.block for region in regions])
    apply_text_evidence(blocks, evidence)
    return blocks


__all__ = ["fill_flash_layout_page"]
