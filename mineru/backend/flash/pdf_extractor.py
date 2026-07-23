"""Flash PDF 原生文本与表格提取器。

Flash 仅处理 PDF 原生文本；需要 OCR 时统一委托 Hybrid low。
"""

from __future__ import annotations

import math
import re
import statistics
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from loguru import logger
from pdftext.schema import Char

from mineru.backend.hybrid.table_text import project_pdf_table_text
from mineru.backend.utils.char_utils import is_hyphen_at_line_end, resolve_text_line_boundary
from mineru.backend.utils.xycut_pp_sorter import sort_entries
from mineru.cli_old.common import read_fn
from mineru.types import BBox
from mineru.utils.language import detect_lang
from mineru.utils.pdf_document import PDFDocument, get_lines_from_chars


_TABLE_CAPTION_RE = re.compile(
    r"^(?:table|tab\.?|表格?)[\s:.–—-]*(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)\b(?P<suffix>.*)$",
    re.IGNORECASE,
)
_NUMERIC_CELL_RE = re.compile(r"(?:\d|%|\bna\b|\bns\b)", re.IGNORECASE)
_TABLE_NOTE_RE = re.compile(
    r"^(?:notes?|sources?)\b|^(?:注释?|说明)\s*[:：]?|^for\s+[*†‡]"
    r"|^(?:[*†‡]|[a-z]|p|t|ns|na)\s+(?:indicates?|denotes?|rainfall\b|total\b|low\b|for\b)",
    re.IGNORECASE,
)
_TABLE_SPLIT_NUMBER_RE = re.compile(
    r"^(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)[.:：]?$",
    re.IGNORECASE,
)
_EQUATION_NUMBER_RE = re.compile(r"^\(\d+(?:\.\d+)*[a-z]?\)$", re.IGNORECASE)
_PDF_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(slots=True)
class _LineItem:
    """保存单个可视文本行及其原始几何信息。"""

    text: str
    bbox: BBox
    angle: int
    source_index: int
    chars: list[Char] = field(default_factory=list)
    visual_row_id: int | None = None
    run_index: int = 0
    effective_height: float = 0.0
    font_signature: tuple[str, int] | None = None
    font_coverage: float = 0.0
    split_from_row: bool = False


@dataclass(slots=True)
class _Fragment:
    """保存表格规则使用的单元文本片段。"""

    text: str
    bbox: BBox
    local_bbox: BBox
    line_index: int
    visual_row_id: int | None = None


@dataclass(slots=True)
class _VisualRow:
    """保存同一局部水平带内的表格片段。"""

    fragments: list[_Fragment]
    center_y: float
    bbox: BBox
    visual_row_id: int | None = None


@dataclass(slots=True)
class _AxisLine:
    """保存 PDF 路径中的横竖线。"""

    bbox: BBox
    width: float
    orientation: Literal["horizontal", "vertical"]


@dataclass(slots=True)
class _LocalAxisLine:
    """保存转入当前文本方向后的横竖线。"""

    bbox: BBox
    original_bbox: BBox
    orientation: Literal["horizontal", "vertical"]
    width: float


@dataclass(slots=True)
class _TableCandidate:
    """保存已通过文本或线框规则的表格候选。"""

    bbox: BBox
    local_bbox: BBox
    angle: int
    score: float
    core_bbox: BBox | None = None
    line_indices: set[int] = field(default_factory=set)
    has_grid: bool = False


@dataclass(slots=True)
class _TextLane:
    """保存同一文本方向下的局部栏带与已归属文本行。"""

    left: float
    right: float
    lines: list[tuple[_LineItem, BBox]] = field(default_factory=list)
    is_span: bool = False


@dataclass(slots=True)
class _PageSource:
    """保存单页原生文本分析所需的文本、字符和绘图线。"""

    page_size: tuple[float, float]
    lines: list[_LineItem]
    chars: list[Char]
    drawing_lines: list[_AxisLine]


def extract_pages_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> list[str]:
    """Extract plain text from each PDF page, preserving empty pages."""

    pages: list[str] = []
    with PDFDocument(filepath) as pdf_doc:
        page_count = pdf_doc.page_count
        end = page_count if end_page is None else min(end_page, page_count)

        # 保留旧 Flash parser 依赖的逐页纯文本接口。
        for page_idx in range(start_page, end):
            pages.append(pdf_doc.get_page_text(page_idx))
    return pages


def _analyze_native_document(pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
    """逐页处理数字 PDF，同一页只读取一次原生字符。"""

    model_list: list[list[dict[str, Any]]] = []
    for page_idx in range(pdf_doc.page_count):
        page_size = pdf_doc.page_size(page_idx)
        chars = pdf_doc.get_page_chars(page_idx)
        lines = _build_native_line_items(
            get_lines_from_chars(chars),
            page_size,
            page_rotation=pdf_doc.page_rotation(page_idx),
        )
        drawing_lines = _get_pdf_drawing_lines(pdf_doc, page_idx)
        source = _PageSource(
            page_size=page_size,
            lines=lines,
            chars=chars,
            drawing_lines=drawing_lines,
        )
        model_list.append(_analyze_page_source(source))
    return model_list


def _build_native_line_items(
    pdf_lines: Sequence[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    page_rotation: int = 0,
) -> list[_LineItem]:
    """将 pdftext 粗行按字符间隙精修成 Flash 视觉 run。"""

    items: list[_LineItem] = []
    for visual_row_id, pdf_line in enumerate(pdf_lines):
        bbox = _clip_bbox(_coerce_bbox(pdf_line.get("bbox")), page_size)
        if bbox is None:
            continue
        spans = pdf_line.get("spans") or []
        chars = [char for span in spans for char in (span.get("chars") or []) if isinstance(char, dict)]
        coarse_item = _LineItem(
            text="".join(str(span.get("text") or "") for span in spans),
            bbox=bbox,
            angle=(_normalize_pdftext_angle(pdf_line.get("rotation")) + int(page_rotation or 0)) % 360,
            source_index=-1,
            chars=chars,
            visual_row_id=visual_row_id,
        )
        items.extend(_split_native_visual_runs(coarse_item, page_size))

    merged_items = _merge_native_inline_scripts(items, page_size)
    for source_index, item in enumerate(merged_items):
        # source_index 必须在页内唯一，表格投影和失败回滚都依赖该精确成员标识。
        item.source_index = source_index
    return merged_items


def _split_native_visual_runs(
    line: _LineItem,
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """保留字符源顺序与空白信息，并将一个 pdftext 粗行拆成远距视觉 run。"""

    tokens: list[tuple[Char, str, BBox | None, BBox | None]] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if raw_char in {"\r", "\n"}:
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle) if bbox is not None else None
        tokens.append((char, raw_char, bbox, local_bbox))

    visible_indices = [
        index
        for index, (_char, raw_char, _bbox, local_bbox) in enumerate(tokens)
        if raw_char.isprintable() and not raw_char.isspace() and local_bbox is not None
    ]
    if not visible_indices:
        text = _normalize_native_run_text(line.text)
        if not text:
            return []
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        line.text = text
        line.effective_height = max(0.1, local_bbox[3] - local_bbox[1])
        return [line]

    glyph_widths = [
        max(0.1, tokens[index][3][2] - tokens[index][3][0])  # type: ignore[index]
        for index in visible_indices
    ]
    median_glyph_width = statistics.median(glyph_widths)
    local_page_width = page_size[1] if line.angle in {90, 270} else page_size[0]
    hard_gap_threshold = max(15.0, 3.0 * median_glyph_width, 0.02 * local_page_width)
    adjacent_gaps: list[float] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        if gap < hard_gap_threshold:
            adjacent_gaps.append(gap)
    # 少于三个相邻样本无法可靠代表“常规”字距；零间隙也是真实的紧排字距，
    # 必须纳入统计，避免唯一的 15pt cell gap 反过来抬高软拆阈值。
    median_regular_gap = statistics.median(adjacent_gaps) if len(adjacent_gaps) >= 3 else 0.0

    split_indices: list[int] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        has_source_whitespace = any(tokens[index][1].isspace() for index in range(previous + 1, current))
        soft_gap_threshold = max(
            8.0,
            2.2 * median_glyph_width,
            3.0 * median_regular_gap,
        )
        if gap >= hard_gap_threshold or (has_source_whitespace and gap >= soft_gap_threshold):
            split_indices.append(current)

    ranges: list[tuple[int, int]] = []
    start = 0
    for split_index in split_indices:
        ranges.append((start, split_index))
        start = split_index
    ranges.append((start, len(tokens)))

    output: list[_LineItem] = []
    for run_index, (start, end) in enumerate(ranges):
        run_tokens = tokens[start:end]
        run_text = _normalize_native_run_text("".join(token[1] for token in run_tokens))
        run_bboxes = [
            token[2]
            for token in run_tokens
            if token[2] is not None and token[1].isprintable() and not token[1].isspace()
        ]
        if not run_text or not run_bboxes:
            continue
        run_chars = [token[0] for token in run_tokens]
        run_item = _LineItem(
            text=run_text,
            bbox=_bbox_union_many(run_bboxes),
            angle=line.angle,
            source_index=-1,
            chars=run_chars,
            visual_row_id=line.visual_row_id,
            run_index=run_index,
            split_from_row=len(ranges) > 1,
        )
        _fill_native_typography(run_item, page_size)
        output.append(run_item)
    return output


def _horizontal_bbox_gap(first_bbox: BBox, second_bbox: BBox) -> float:
    """返回两个局部 bbox 在 x 轴上的无方向净空，重叠时为零。"""

    return max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)


def _normalize_native_run_text(text: str) -> str:
    """清理原生 run 文本，并把字母后的 PDF 软断词标记转换成 ASCII hyphen。"""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=[A-Za-z])\x02(?=\s*$)", "-", normalized)
    normalized = _sanitize_pdf_control_text(normalized, preserve_newlines=False)
    normalized = re.sub(r"[\t\f\v ]+", " ", normalized)
    return normalized.strip()


def _sanitize_pdf_control_text(text: str, *, preserve_newlines: bool) -> str:
    """删除 PDF 字体编码残留控制字符，并按调用场景决定是否保留物理换行。"""

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=[A-Za-z])\x02(?=[\t ]*(?:\n|$))", "-", normalized)
    normalized = normalized.replace("\t", " ")
    if not preserve_newlines:
        normalized = normalized.replace("\n", "")
    return _PDF_CONTROL_CHAR_RE.sub("", normalized)


def _fill_native_typography(line: _LineItem, page_size: tuple[float, float]) -> None:
    """使用非空字符 bbox 高度和 dominant font 填充原生行排版特征。"""

    heights: list[float] = []
    font_counts: dict[tuple[str, int], int] = {}
    valid_font_chars = 0
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if not raw_char.isprintable() or raw_char.isspace():
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        heights.append(max(0.1, local_bbox[3] - local_bbox[1]))
        font = char.get("font") or {}
        font_name = str(font.get("name") or "")
        if not font_name:
            continue
        try:
            font_flags = int(font.get("flags") or 0)
        except (TypeError, ValueError):
            font_flags = 0
        signature = (font_name, font_flags)
        font_counts[signature] = font_counts.get(signature, 0) + 1
        valid_font_chars += 1

    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    line.effective_height = statistics.median(heights) if heights else max(0.1, local_bbox[3] - local_bbox[1])
    if font_counts and valid_font_chars:
        line.font_signature, dominant_count = max(font_counts.items(), key=lambda item: item[1])
        line.font_coverage = dominant_count / valid_font_chars
    else:
        line.font_signature = None
        line.font_coverage = 0.0


def _merge_native_inline_scripts(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """以 mutual-nearest 规则把跨粗行的小字号前后置标记合入主体视觉行。"""

    candidates: list[tuple[float, int, int, Literal["prefix", "suffix"]]] = []
    for small_index, small in enumerate(lines):
        compact_text = "".join(char for char in small.text if not char.isspace())
        if not compact_text:
            continue
        small_local_bbox = _rotate_bbox_to_upright(small.bbox, page_size, small.angle)
        for base_index, base in enumerate(lines):
            if small_index == base_index or small.angle != base.angle or small.visual_row_id == base.visual_row_id:
                continue
            if small.effective_height <= 0 or base.effective_height <= 0:
                continue
            height_ratio = small.effective_height / base.effective_height
            if not 0.35 <= height_ratio <= 0.8:
                continue
            if len(compact_text) > 8 and small_local_bbox[2] - small_local_bbox[0] > 3.0 * base.effective_height:
                continue
            base_local_bbox = _rotate_bbox_to_upright(base.bbox, page_size, base.angle)
            vertical_overlap = max(
                0.0,
                min(small_local_bbox[3], base_local_bbox[3]) - max(small_local_bbox[1], base_local_bbox[1]),
            )
            small_height = max(0.1, small_local_bbox[3] - small_local_bbox[1])
            overlap_ratio = vertical_overlap / small_height
            if overlap_ratio < 0.5:
                continue
            center_offset = abs(_bbox_center_y(small_local_bbox) - _bbox_center_y(base_local_bbox))
            if center_offset < max(0.5, 0.12 * base.effective_height):
                # 同基线居中的小字号文本更可能是表格相邻 cell，而不是上下标。
                continue
            outside_offset = max(
                base_local_bbox[1] - small_local_bbox[1],
                small_local_bbox[3] - base_local_bbox[3],
            )
            if outside_offset < max(0.5, 0.08 * base.effective_height):
                # 普通小字号 cell 即使底边与主体对齐，也不能只凭中心偏移冒充上下标。
                continue

            if _bbox_center_x(small_local_bbox) < _bbox_center_x(base_local_bbox):
                position: Literal["prefix", "suffix"] = "prefix"
                gap = base_local_bbox[0] - small_local_bbox[2]
            else:
                position = "suffix"
                gap = small_local_bbox[0] - base_local_bbox[2]
            if not -0.25 * base.effective_height <= gap <= max(1.5, 0.35 * base.effective_height):
                continue
            metric = abs(gap) + (1.0 - overlap_ratio) * base.effective_height
            candidates.append((metric, small_index, base_index, position))

    best_base_for_small: dict[int, tuple[float, int, Literal["prefix", "suffix"]]] = {}
    best_small_for_base: dict[tuple[int, str], tuple[float, int]] = {}
    for metric, small_index, base_index, position in candidates:
        if small_index not in best_base_for_small or metric < best_base_for_small[small_index][0]:
            best_base_for_small[small_index] = (metric, base_index, position)
        base_key = (base_index, position)
        if base_key not in best_small_for_base or metric < best_small_for_base[base_key][0]:
            best_small_for_base[base_key] = (metric, small_index)

    matches: dict[int, dict[str, int]] = {}
    for small_index, (_metric, base_index, position) in best_base_for_small.items():
        if best_small_for_base.get((base_index, position), (math.inf, -1))[1] == small_index:
            matches.setdefault(base_index, {})[position] = small_index

    consumed_small_indices = {small_index for positions in matches.values() for small_index in positions.values()}
    merged_base_indices: set[int] = set()

    def merge_children(base_index: int, visiting: set[int]) -> None:
        """先合并更小的依赖标记，再把当前完整节点递归合入更大的主体行。"""

        if base_index in merged_base_indices or base_index in visiting:
            return
        visiting.add(base_index)
        positions = matches.get(base_index, {})
        for child_index in positions.values():
            merge_children(child_index, visiting)
        base = lines[base_index]
        merged_bbox = base.bbox
        merged_chars = list(base.chars)
        if "prefix" in positions:
            prefix_index = positions["prefix"]
            prefix = lines[prefix_index]
            base.text = f"{prefix.text.strip()} {base.text.lstrip()}"
            merged_bbox = _bbox_union(merged_bbox, prefix.bbox)
            merged_chars = [*prefix.chars, *merged_chars]
            base.split_from_row = base.split_from_row or prefix.split_from_row
        if "suffix" in positions:
            suffix_index = positions["suffix"]
            suffix = lines[suffix_index]
            base.text = f"{base.text.rstrip()}{suffix.text.strip()}"
            merged_bbox = _bbox_union(merged_bbox, suffix.bbox)
            merged_chars.extend(suffix.chars)
            base.split_from_row = base.split_from_row or suffix.split_from_row
        base.bbox = merged_bbox
        base.chars = merged_chars
        _fill_native_typography(base, page_size)
        visiting.remove(base_index)
        merged_base_indices.add(base_index)

    # 从最终不会被消费的根主体开始，确保 small -> medium -> large 链不会丢失最小节点。
    root_base_indices = [base_index for base_index in matches if base_index not in consumed_small_indices]
    for base_index in root_base_indices:
        merge_children(base_index, set())
    for base_index in matches:
        merge_children(base_index, set())

    output = [line for index, line in enumerate(lines) if index not in consumed_small_indices]
    output.sort(key=lambda item: (item.visual_row_id if item.visual_row_id is not None else math.inf, item.run_index))
    return output


def _normalize_pdftext_angle(value: Any) -> int:
    """将 pdftext 弧度方向就近归一到四个标准角度。"""

    try:
        angle_radians = float(value or 0.0)
    except (TypeError, ValueError):
        return 0
    angle_degrees = math.degrees(angle_radians)
    normalized = int(round(angle_degrees / 90.0) * 90) % 360
    return normalized if normalized in {0, 90, 180, 270} else 0


def _get_pdf_drawing_lines(pdf_doc: PDFDocument, page_idx: int) -> list[_AxisLine]:
    """读取 PDFDocument 的公共绘图线结果，并隔离具体 PDFium 类型。"""

    output: list[_AxisLine] = []
    for drawing_line in pdf_doc.get_page_drawing_lines(page_idx):
        bbox = _coerce_bbox(drawing_line.bbox)
        if bbox is None:
            continue
        output.append(
            _AxisLine(
                bbox=bbox,
                width=max(0.0, float(drawing_line.width)),
                orientation=drawing_line.orientation,
            )
        )
    return output


def _analyze_page_source(source: _PageSource) -> list[dict[str, Any]]:
    """在单页内先确认表格，再聚合剩余文本并排序。"""

    if not source.lines:
        return []
    candidates = _detect_table_candidates(source)
    table_blocks, claimed_line_indices = _materialize_table_blocks(
        source,
        candidates,
    )
    table_bboxes = [block["bbox"] for block in table_blocks]
    remaining_lines = _merge_same_baseline_text_lines(
        [line for line in source.lines if line.source_index not in claimed_line_indices],
        source.page_size,
        table_bboxes,
    )
    formula_blocks, remaining_lines = _build_formula_like_blocks(
        remaining_lines,
        table_bboxes,
        source.page_size,
    )
    remaining_lines = _restore_dense_split_visual_rows(
        remaining_lines,
        source.page_size,
        table_bboxes,
    )
    text_blocks = _build_text_blocks(
        remaining_lines,
        table_bboxes,
        source.page_size,
        source.drawing_lines,
    )
    absolute_blocks = table_blocks + formula_blocks + text_blocks
    sorted_blocks = _sort_blocks_with_visual_row_groups(absolute_blocks, source.page_size)
    return [
        normalized for block in sorted_blocks if (normalized := _normalize_output_block(block, source.page_size)) is not None
    ]


def _detect_table_candidates(source: _PageSource) -> list[_TableCandidate]:
    """融合多列文本稳定性与横竖线证据，生成高精度表格候选。"""

    candidates: list[_TableCandidate] = []
    angles = sorted({line.angle for line in source.lines})
    for angle in angles:
        angle_lines = [line for line in source.lines if line.angle == angle]
        if not angle_lines:
            continue
        fragments = _build_fragments(angle_lines, source.page_size)
        if not fragments:
            continue
        median_height = _median_fragment_height(fragments)
        rows = _cluster_fragment_rows(fragments, median_height)
        local_axis_lines = _transform_axis_lines(
            source.drawing_lines,
            source.page_size,
            angle,
        )
        candidates.extend(
            _build_text_table_candidates(
                rows,
                angle_lines,
                source.page_size,
                angle,
                median_height,
                local_axis_lines,
            )
        )
        candidates.extend(
            _build_grid_table_candidates(
                fragments,
                rows,
                angle_lines,
                source.page_size,
                angle,
                median_height,
                local_axis_lines,
            )
        )
    return _merge_table_candidates(candidates)


def _build_fragments(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_Fragment]:
    """将精修后的原生 run 转换成表格单元候选。"""

    fragments: list[_Fragment] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        fragments.append(
            _Fragment(
                text=line.text,
                bbox=line.bbox,
                local_bbox=local_bbox,
                line_index=line.source_index,
                # 复用原生粗行身份，避免同一字符行内不同 cell
                # 因轻微基线差异被误拆成多行。
                visual_row_id=line.visual_row_id,
            )
        )
    return fragments


def _cluster_fragment_rows(
    fragments: list[_Fragment],
    median_height: float,
) -> list[_VisualRow]:
    """优先复用原生视觉行身份，其余片段按中心线容差聚成表格行。"""

    tolerance = max(2.0, median_height * 0.5)
    native_groups: dict[int, list[_Fragment]] = {}
    geometric_fragments: list[_Fragment] = []
    for fragment in fragments:
        if fragment.visual_row_id is None:
            geometric_fragments.append(fragment)
        else:
            native_groups.setdefault(fragment.visual_row_id, []).append(fragment)

    # 先锁定同一原生粗行拆出的 run，再允许不同粗行按基线几何合并；
    # 旋转表格常把同一数据行的各 cell 分成多个 pdftext 粗行，不能只依赖 row id。
    seed_groups = [*native_groups.values(), *[[fragment] for fragment in geometric_fragments]]
    seed_groups.sort(
        key=lambda group: (
            statistics.fmean(_bbox_center_y(item.local_bbox) for item in group),
            min(item.local_bbox[0] for item in group),
        )
    )
    grouped: list[list[_Fragment]] = []
    for seed_group in seed_groups:
        center_y = statistics.fmean(_bbox_center_y(item.local_bbox) for item in seed_group)
        target_group: list[_Fragment] | None = None
        for group in grouped:
            group_center = statistics.fmean(_bbox_center_y(item.local_bbox) for item in group)
            if abs(center_y - group_center) <= tolerance:
                target_group = group
                break
        if target_group is None:
            grouped.append(list(seed_group))
        else:
            target_group.extend(seed_group)

    rows: list[_VisualRow] = []
    for group in grouped:
        group.sort(key=lambda item: item.local_bbox[0])
        bbox = _bbox_union_many([item.local_bbox for item in group])
        visual_row_ids = {item.visual_row_id for item in group if item.visual_row_id is not None}
        rows.append(
            _VisualRow(
                fragments=group,
                center_y=sum(_bbox_center_y(item.local_bbox) for item in group) / len(group),
                bbox=bbox,
                visual_row_id=next(iter(visual_row_ids)) if len(visual_row_ids) == 1 else None,
            )
        )
    rows.sort(key=lambda row: row.center_y)
    return rows


def _build_text_table_candidates(
    rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
) -> list[_TableCandidate]:
    """从连续多列行中识别无边框或横线型表格。"""

    candidate_rows = [row for row in rows if len(row.fragments) >= 2]
    if len(candidate_rows) < 3:
        return []
    max_gap = max(4.0 * median_height, 8.0)
    segments: list[list[_VisualRow]] = []
    for row in candidate_rows:
        if not segments or row.center_y - segments[-1][-1].center_y > max_gap:
            segments.append([row])
        else:
            segments[-1].append(row)

    candidates: list[_TableCandidate] = []
    for segment in segments:
        if len(segment) < 3:
            continue
        core_bbox = _bbox_union_many([row.bbox for row in segment])
        caption_line = _find_table_caption(lines, core_bbox, page_size, angle, median_height)
        if caption_line is None and _looks_like_numbered_equation_segment(segment, core_bbox):
            continue
        stable_columns, column_coverage = _count_stable_columns(segment, median_height)
        multi_cell_ratio = sum(len(row.fragments) >= 3 for row in segment) / len(segment)
        accepts_three_columns = stable_columns >= 3 and multi_cell_ratio >= 0.6
        accepts_two_columns = (
            stable_columns == 2
            and column_coverage >= 0.7
            and len(segment) >= 5
            and (caption_line is not None or _two_column_data_row_ratio(segment) >= 0.5)
        )
        if not accepts_three_columns and not accepts_two_columns:
            continue

        candidate = _expand_text_candidate(
            segment,
            lines,
            page_size,
            angle,
            median_height,
            caption_line,
        )
        horizontal_count, vertical_count, intersection_count = _candidate_grid_evidence(
            candidate.local_bbox,
            axis_lines,
            median_height,
        )
        candidate.has_grid = horizontal_count >= 2 and vertical_count >= 2 and intersection_count >= 2
        horizontal_rule_evidence = horizontal_count >= 3 and stable_columns >= 3
        candidate.score = (
            stable_columns * 2.0
            + len(segment)
            + multi_cell_ratio * 3.0
            + (4.0 if candidate.has_grid else 0.0)
            + (2.0 if horizontal_rule_evidence else 0.0)
            + (1.0 if caption_line is not None else 0.0)
        )
        candidates.append(candidate)
    return candidates


def _looks_like_numbered_equation_segment(
    rows: list[_VisualRow],
    core_bbox: BBox,
) -> bool:
    """识别短行组右侧的孤立公式编号，避免把多层公式误判为无框表格。"""

    if len(rows) > 6:
        return False
    core_width = max(0.1, core_bbox[2] - core_bbox[0])
    right_threshold = core_bbox[0] + 0.75 * core_width
    return any(
        _EQUATION_NUMBER_RE.match(fragment.text.strip())
        and _bbox_center_x(fragment.local_bbox) >= right_threshold
        for row in rows
        for fragment in row.fragments
    )


def _expand_text_candidate(
    core_rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    caption_line: _LineItem | None,
) -> _TableCandidate:
    """将核心数据行向上扩展到标题和表头，向下吸收近邻注释。"""

    core_local_bbox = _bbox_union_many([row.bbox for row in core_rows])
    local_bottom = core_local_bbox[3]
    included: list[_LineItem] = []
    core_line_indices = {fragment.line_index for row in core_rows for fragment in row.fragments}
    bridge_bottom = local_bottom
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        center_y = _bbox_center_y(local_bbox)
        horizontal_overlap = _bbox_axis_overlap_ratio(local_bbox, core_local_bbox, axis="x")
        if line.source_index in core_line_indices or (
            core_local_bbox[1] <= center_y <= local_bottom and horizontal_overlap >= 0.05
        ):
            included.append(line)

    # 仅在核心表格上方两倍行高内吸收表头；远距离显式标题单独加入，不能吞掉桥接走廊正文。
    # 表格注释可以有多个物理行，按相邻行间隙逐行向下扩展。
    included_indices = {line.source_index for line in included}
    upper_lines = sorted(
        (
            line
            for line in lines
            if line.source_index not in included_indices
            and _rotate_bbox_to_upright(line.bbox, page_size, angle)[3] <= core_local_bbox[1]
        ),
        key=lambda line: _rotate_bbox_to_upright(line.bbox, page_size, angle)[3],
        reverse=True,
    )
    for line in upper_lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        if core_local_bbox[1] - local_bbox[3] > 2.0 * median_height:
            break
        horizontal_overlap = _bbox_axis_overlap_ratio(local_bbox, core_local_bbox, axis="x")
        if horizontal_overlap >= 0.05:
            included.append(line)
            included_indices.add(line.source_index)

    caption_members = (
        _table_caption_members(caption_line, lines, page_size, angle, median_height) if caption_line is not None else []
    )
    for caption_member in caption_members:
        if caption_member.source_index not in included_indices:
            included.append(caption_member)
            included_indices.add(caption_member.source_index)

    if caption_members:
        caption_local_bbox = _bbox_union_many(
            [_rotate_bbox_to_upright(member.bbox, page_size, angle) for member in caption_members]
        )
        caption_bridge_bottom = caption_local_bbox[3]
        caption_member_indices = {member.source_index for member in caption_members}
        bridge_lines = sorted(
            (
                line
                for line in lines
                if line.source_index not in caption_member_indices
                and caption_local_bbox[1] <= _rotate_bbox_to_upright(line.bbox, page_size, angle)[1] <= core_local_bbox[1]
            ),
            key=lambda line: _rotate_bbox_to_upright(line.bbox, page_size, angle)[1],
        )
        # 显式标题与表格核心之间仅沿不超过两倍行高的连续文本链桥接，避免吸收大空隙后的正文。
        pending_bridge_lines: list[_LineItem] = []
        for line in bridge_lines:
            local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
            gap = local_bbox[1] - caption_bridge_bottom
            if gap > 2.0 * median_height:
                break
            horizontal_overlap = _bbox_axis_overlap_ratio(local_bbox, core_local_bbox, axis="x")
            if gap >= -median_height and horizontal_overlap >= 0.05:
                if line.source_index not in included_indices:
                    pending_bridge_lines.append(line)
                caption_bridge_bottom = max(caption_bridge_bottom, local_bbox[3])
        if core_local_bbox[1] - caption_bridge_bottom <= 2.0 * median_height:
            included.extend(pending_bridge_lines)
            included_indices.update(line.source_index for line in pending_bridge_lines)

    lower_lines = sorted(
        (
            line
            for line in lines
            if line.source_index not in included_indices
            and _rotate_bbox_to_upright(line.bbox, page_size, angle)[1] >= local_bottom
        ),
        key=lambda line: _rotate_bbox_to_upright(line.bbox, page_size, angle)[1],
    )
    note_chain_started = False
    for line in lower_lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        horizontal_overlap = _bbox_axis_overlap_ratio(local_bbox, core_local_bbox, axis="x")
        gap = local_bbox[1] - bridge_bottom
        if gap > 2.0 * median_height:
            break
        if gap >= -median_height and horizontal_overlap >= 0.05:
            if not note_chain_started and not _is_table_note_text(line.text):
                break
            note_chain_started = True
            included.append(line)
            included_indices.add(line.source_index)
            bridge_bottom = max(bridge_bottom, local_bbox[3])

    if not included:
        included = [line for line in lines if line.source_index in core_line_indices]
    original_bbox = _bbox_union_many([line.bbox for line in included])
    local_bbox = _bbox_union_many([_rotate_bbox_to_upright(line.bbox, page_size, angle) for line in included])
    return _TableCandidate(
        bbox=original_bbox,
        local_bbox=local_bbox,
        angle=angle,
        score=0.0,
        core_bbox=_rotate_bbox_from_upright(core_local_bbox, page_size, angle),
        line_indices={line.source_index for line in included},
    )


def _is_table_note_text(text: str) -> bool:
    """判断表后首行是否具有明确的注释、来源或脚注标记。"""

    return bool(_TABLE_NOTE_RE.match(str(text or "").strip()))


def _find_table_caption(
    lines: list[_LineItem],
    core_bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> _LineItem | None:
    """在核心表格上方最多十二倍行高内查找显式 Table/表标题。"""

    candidates: list[tuple[float, _LineItem]] = []
    for line in lines:
        text = line.text.strip()
        caption_match = _TABLE_CAPTION_RE.match(text)
        is_split_label = text.lower().rstrip(".") in {"table", "tab", "表", "表格"}
        if caption_match is None and not is_split_label:
            continue
        if caption_match is not None:
            suffix = caption_match.group("suffix").strip(" .:–—-")
            # 小写连续句通常是“Table 5 also ...”这类正文，不应作为标题。
            if suffix and suffix[0].islower():
                continue
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        if is_split_label:
            has_number_peer = bool(
                _find_caption_number_peers(
                    line,
                    lines,
                    page_size,
                    angle,
                    median_height,
                )
            )
            if not has_number_peer:
                continue
        gap = core_bbox[1] - local_bbox[3]
        if -median_height <= gap <= 12.0 * median_height:
            candidates.append((abs(gap), line))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _find_caption_number_peers(
    caption_line: _LineItem,
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> list[_LineItem]:
    """查找与拆分 Table/表 标签同一视觉行的编号文本。"""

    caption_local_bbox = _rotate_bbox_to_upright(caption_line.bbox, page_size, angle)
    peers: list[_LineItem] = []
    for peer in lines:
        if peer.source_index == caption_line.source_index:
            continue
        if not _TABLE_SPLIT_NUMBER_RE.match(peer.text.strip()):
            continue
        peer_local_bbox = _rotate_bbox_to_upright(peer.bbox, page_size, angle)
        gap = peer_local_bbox[0] - caption_local_bbox[2]
        if _bbox_axis_overlap_ratio(caption_local_bbox, peer_local_bbox, axis="y") >= 0.5 and 0.0 <= gap <= 4.0 * median_height:
            peers.append(peer)
    return sorted(
        peers,
        key=lambda peer: _rotate_bbox_to_upright(peer.bbox, page_size, angle)[0],
    )


def _table_caption_members(
    caption_line: _LineItem,
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> list[_LineItem]:
    """返回显式表标题标签及其可能被 pdftext 拆开的编号成员。"""

    text = caption_line.text.strip().lower().rstrip(".")
    if text not in {"table", "tab", "表", "表格"}:
        return [caption_line]
    return [
        caption_line,
        *_find_caption_number_peers(
            caption_line,
            lines,
            page_size,
            angle,
            median_height,
        ),
    ]


def _count_stable_columns(
    rows: list[_VisualRow],
    median_height: float,
) -> tuple[int, float]:
    """聚类每行片段左边界，返回稳定列数和最低列覆盖率。"""

    tolerance = max(3.0, median_height * 0.75)
    clusters: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for fragment in row.fragments:
            anchor = fragment.local_bbox[0]
            cluster = next(
                (item for item in clusters if abs(anchor - float(item["mean"])) <= tolerance),
                None,
            )
            if cluster is None:
                clusters.append({"mean": anchor, "values": [anchor], "rows": {row_index}})
            else:
                cluster["values"].append(anchor)
                cluster["rows"].add(row_index)
                cluster["mean"] = sum(cluster["values"]) / len(cluster["values"])
    stable_coverages = [len(cluster["rows"]) / len(rows) for cluster in clusters if len(cluster["rows"]) / len(rows) >= 0.5]
    if not stable_coverages:
        return 0, 0.0
    return len(stable_coverages), min(stable_coverages)


def _two_column_data_row_ratio(rows: list[_VisualRow]) -> float:
    """统计两列候选中两个单元都具有短值或数值特征的行比例。"""

    if not rows:
        return 0.0
    supported_rows = 0
    for row in rows:
        if len(row.fragments) != 2:
            continue
        if all(len(fragment.text.strip()) <= 20 or bool(_NUMERIC_CELL_RE.search(fragment.text)) for fragment in row.fragments):
            supported_rows += 1
    return supported_rows / len(rows)


def _transform_axis_lines(
    lines: list[_AxisLine],
    page_size: tuple[float, float],
    angle: int,
) -> list[_LocalAxisLine]:
    """将原页面横竖线转入当前文本方向的局部坐标。"""

    output: list[_LocalAxisLine] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        orientation: Literal["horizontal", "vertical"] = (
            "horizontal" if local_bbox[2] - local_bbox[0] >= local_bbox[3] - local_bbox[1] else "vertical"
        )
        output.append(
            _LocalAxisLine(
                bbox=local_bbox,
                original_bbox=line.bbox,
                orientation=orientation,
                width=line.width,
            )
        )
    return output


def _candidate_grid_evidence(
    candidate_bbox: BBox,
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> tuple[int, int, int]:
    """统计候选内长横线、竖线和交点数。"""

    margin = 2.0 * median_height
    expanded = _expand_bbox(candidate_bbox, margin)
    selected = [line for line in axis_lines if _bbox_intersects(line.bbox, expanded)]
    horizontal = [
        line
        for line in selected
        if line.orientation == "horizontal" and line.bbox[2] - line.bbox[0] >= max(20.0, median_height * 4.0)
    ]
    vertical = [
        line
        for line in selected
        if line.orientation == "vertical" and line.bbox[3] - line.bbox[1] >= max(20.0, median_height * 4.0)
    ]
    intersections = sum(_axis_lines_intersect(h_line, v_line, tolerance=2.0) for h_line in horizontal for v_line in vertical)
    return len(horizontal), len(vertical), intersections


def _build_grid_table_candidates(
    fragments: list[_Fragment],
    rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
) -> list[_TableCandidate]:
    """对相交横竖线连通分量生成独立强网格表格候选。"""

    components = _axis_line_components(axis_lines, tolerance=2.0)
    candidates: list[_TableCandidate] = []
    for component in components:
        horizontal = [line for line in component if line.orientation == "horizontal"]
        vertical = [line for line in component if line.orientation == "vertical"]
        if len(horizontal) < 2 or len(vertical) < 2:
            continue
        intersection_count = sum(
            _axis_lines_intersect(h_line, v_line, tolerance=2.0) for h_line in horizontal for v_line in vertical
        )
        if intersection_count < 4:
            continue
        local_bbox = _bbox_union_many([line.bbox for line in component])
        selected_fragments = [
            fragment
            for fragment in fragments
            if _point_in_bbox(
                (_bbox_center_x(fragment.local_bbox), _bbox_center_y(fragment.local_bbox)),
                local_bbox,
            )
        ]
        selected_line_indices = {fragment.line_index for fragment in selected_fragments}
        selected_rows = [row for row in rows if any(fragment.line_index in selected_line_indices for fragment in row.fragments)]
        if len(selected_rows) < 2:
            continue
        grid_text_columns = _count_grid_text_columns(selected_fragments, vertical)
        # 强网格按竖线分隔后的实际文本占位计列，避免字符左边界波动时漏掉有框表格。
        if grid_text_columns < 2:
            continue
        original_bbox = _rotate_bbox_from_upright(local_bbox, page_size, angle)
        candidate = _TableCandidate(
            bbox=original_bbox,
            local_bbox=local_bbox,
            angle=angle,
            score=20.0 + intersection_count + len(selected_rows),
            core_bbox=original_bbox,
            line_indices=selected_line_indices,
            has_grid=True,
        )
        caption = _find_table_caption(lines, local_bbox, page_size, angle, median_height)
        if caption is not None:
            for caption_member in _table_caption_members(
                caption,
                lines,
                page_size,
                angle,
                median_height,
            ):
                candidate.bbox = _bbox_union(candidate.bbox, caption_member.bbox)
                candidate.local_bbox = _bbox_union(
                    candidate.local_bbox,
                    _rotate_bbox_to_upright(caption_member.bbox, page_size, angle),
                )
                candidate.line_indices.add(caption_member.source_index)
        candidates.append(candidate)
    return candidates


def _count_grid_text_columns(
    fragments: list[_Fragment],
    vertical_lines: list[_LocalAxisLine],
) -> int:
    """按竖线中心形成的区间统计实际包含文本的网格列数。"""

    if not fragments or not vertical_lines:
        return 0
    boundaries: list[float] = []
    for coordinate in sorted(_bbox_center_x(line.bbox) for line in vertical_lines):
        if not boundaries or coordinate - boundaries[-1] > 2.0:
            boundaries.append(coordinate)
    occupied_columns = {bisect_right(boundaries, _bbox_center_x(fragment.local_bbox)) for fragment in fragments}
    return len(occupied_columns)


def _axis_line_components(
    lines: list[_LocalAxisLine],
    tolerance: float,
) -> list[list[_LocalAxisLine]]:
    """将相交或相邻的横竖线聚成网格连通分量。"""

    if not lines:
        return []
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找线段所属并查集根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        """合并两条相交或相邻线段的连通分量。"""

        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_line in enumerate(lines):
        for right_index in range(left_index + 1, len(lines)):
            right_line = lines[right_index]
            if _bbox_intersects(
                _expand_bbox(left_line.bbox, tolerance),
                _expand_bbox(right_line.bbox, tolerance),
            ):
                union(left_index, right_index)

    groups: dict[int, list[_LocalAxisLine]] = {}
    for index, line in enumerate(lines):
        groups.setdefault(find(index), []).append(line)
    return list(groups.values())


def _merge_table_candidates(candidates: list[_TableCandidate]) -> list[_TableCandidate]:
    """合并同方向且明显重叠的文本/网格候选，避免同一表格重复输出。"""

    merged: list[_TableCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        target = next(
            (
                item
                for item in merged
                if item.angle == candidate.angle and _bbox_overlap_in_smaller(candidate.bbox, item.bbox) >= 0.2
            ),
            None,
        )
        if target is None:
            merged.append(candidate)
            continue
        target.bbox = _bbox_union(target.bbox, candidate.bbox)
        target.local_bbox = _bbox_union(target.local_bbox, candidate.local_bbox)
        if target.core_bbox is None:
            target.core_bbox = candidate.core_bbox
        elif candidate.core_bbox is not None:
            target.core_bbox = _bbox_union(target.core_bbox, candidate.core_bbox)
        target.line_indices.update(candidate.line_indices)
        target.score = max(target.score, candidate.score)
        target.has_grid = target.has_grid or candidate.has_grid
    return sorted(merged, key=lambda item: (item.bbox[1], item.bbox[0]))


def _materialize_table_blocks(
    source: _PageSource,
    candidates: list[_TableCandidate],
) -> tuple[list[dict[str, Any]], set[int]]:
    """为候选生成空间投影 content，仅认领投影成功的文本行。"""

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_bbox_overlap_in_smaller(candidate.bbox, block["bbox"]) >= 0.5 for block in blocks):
            continue
        output_angle = candidate.angle
        projection_line_indices = _candidate_projection_line_indices(source, candidate)
        try:
            candidate_chars = [
                char for line in source.lines if line.source_index in projection_line_indices for char in line.chars
            ]
            content = project_pdf_table_text(
                candidate_chars,
                candidate.bbox,
                angle=candidate.angle,
            )
        except Exception as exc:
            # 单个表格的字符投影异常只撤销该候选，不能中止整页提取。
            logger.warning(f"Flash table projection failed and rolled back: bbox={candidate.bbox}, error={exc}")
            continue
        if not content or not content.strip():
            continue
        blocks.append(
            {
                "type": "table",
                "bbox": candidate.bbox,
                "angle": output_angle,
                "content": content,
            }
        )
        # 只认领候选明确接纳的视觉行，避免远标题与表格之间的正文被矩形 bbox 连带删除。
        claimed.update(projection_line_indices)
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _candidate_projection_line_indices(
    source: _PageSource,
    candidate: _TableCandidate,
) -> set[int]:
    """合并核心成员、同基线续段及非零角度表格的表头文本。"""

    line_indices = set(candidate.line_indices)
    if candidate.core_bbox is not None:
        for line in source.lines:
            if _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                candidate.core_bbox,
            ):
                line_indices.add(line.source_index)
    if candidate.angle == 0:
        _expand_candidate_same_baseline_members(source, candidate, line_indices)
        return line_indices
    if candidate.core_bbox is None:
        return line_indices

    candidate_local_bbox = _rotate_bbox_to_upright(
        candidate.bbox,
        source.page_size,
        candidate.angle,
    )
    core_local_bbox = _rotate_bbox_to_upright(
        candidate.core_bbox,
        source.page_size,
        candidate.angle,
    )
    for line in source.lines:
        if line.angle != candidate.angle:
            continue
        page_center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        if not _point_in_bbox(page_center, candidate.bbox):
            continue
        local_bbox = _rotate_bbox_to_upright(
            line.bbox,
            source.page_size,
            candidate.angle,
        )
        local_center_y = _bbox_center_y(local_bbox)
        if not candidate_local_bbox[1] <= local_center_y <= candidate_local_bbox[3]:
            continue
        if _bbox_axis_overlap_ratio(local_bbox, core_local_bbox, axis="x") < 0.05:
            continue
        line_indices.add(line.source_index)
    return line_indices


def _expand_candidate_same_baseline_members(
    source: _PageSource,
    candidate: _TableCandidate,
    line_indices: set[int],
) -> None:
    """迭代吸收完整候选框内与已认领成员同基线相邻的 angle=0 续段。"""

    local_bboxes = {
        line.source_index: _rotate_bbox_to_upright(
            line.bbox,
            source.page_size,
            candidate.angle,
        )
        for line in source.lines
        if line.angle == candidate.angle
    }
    changed = True
    while changed:
        changed = False
        selected_lines = [
            line
            for line in source.lines
            if line.angle == candidate.angle and line.source_index in line_indices
        ]
        for line in source.lines:
            if line.angle != candidate.angle or line.source_index in line_indices:
                continue
            if not _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                candidate.bbox,
            ):
                continue
            line_bbox = local_bboxes[line.source_index]
            line_height = _line_effective_height(line, line_bbox)
            for selected in selected_lines:
                if (
                    line.font_signature is not None
                    and selected.font_signature is not None
                    and line.font_signature != selected.font_signature
                ):
                    continue
                selected_bbox = local_bboxes[selected.source_index]
                if not _same_baseline_geometry(
                    line_bbox,
                    line_height,
                    selected_bbox,
                    _line_effective_height(selected, selected_bbox),
                ):
                    continue
                line_indices.add(line.source_index)
                changed = True
                break


def _merge_same_baseline_text_lines(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在表格认领后合并同基线、同字体且水平邻近的正文 run。"""

    if len(lines) < 2:
        return list(lines)
    local_bboxes = [_rotate_bbox_to_upright(line.bbox, page_size, line.angle) for line in lines]
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找同行合并并查集的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        """合并两个满足同行条件的文本 run。"""

        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_line in enumerate(lines):
        for right_index in range(left_index + 1, len(lines)):
            right_line = lines[right_index]
            if _can_merge_same_baseline_pair(
                left_line,
                local_bboxes[left_index],
                right_line,
                local_bboxes[right_index],
                table_bboxes,
            ):
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)

    output: list[_LineItem] = []
    for indices in groups.values():
        if len(indices) == 1:
            output.append(lines[indices[0]])
            continue
        indices.sort(key=lambda index: (local_bboxes[index][0], local_bboxes[index][1], lines[index].source_index))
        output.append(_merge_same_baseline_group(indices, lines, local_bboxes, page_size))
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _can_merge_same_baseline_pair(
    first: _LineItem,
    first_bbox: BBox,
    second: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """判断两个剩余文本 run 是否属于同一条物理基线。"""

    if first.angle != second.angle:
        return False
    if first.visual_row_id == second.visual_row_id and (first.split_from_row or second.split_from_row):
        return False
    if (
        first.font_signature is None
        or second.font_signature is None
        or first.font_coverage < 0.75
        or second.font_coverage < 0.75
        or first.font_signature != second.font_signature
    ):
        return False
    first_height = _line_effective_height(first, first_bbox)
    second_height = _line_effective_height(second, second_bbox)
    if not _same_baseline_geometry(
        first_bbox,
        first_height,
        second_bbox,
        second_height,
    ):
        return False
    return not _connection_crosses_table(first.bbox, second.bbox, table_bboxes)


def _same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
    *,
    maximum_gap: float | None = None,
) -> bool:
    """仅依据行高、垂直交叠和水平净空判断两个局部 bbox 是否同基线相邻。"""

    pair_height = max(first_height, second_height)
    if min(first_height, second_height) <= 0 or pair_height / min(first_height, second_height) > 1.35:
        return False
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    shorter_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / shorter_bbox_height < 0.7 and abs(first_bbox[3] - second_bbox[3]) > 0.25 * pair_height:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    signed_gap = right_bbox[0] - left_bbox[2]
    gap_limit = max(3.0, 0.75 * pair_height) if maximum_gap is None else maximum_gap
    return -0.25 * pair_height <= signed_gap <= gap_limit


def _merge_same_baseline_group(
    indices: list[int],
    lines: list[_LineItem],
    local_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序合并一个同基线分量，并保留全部字符与几何信息。"""

    members = [lines[index] for index in indices]
    content_parts = [members[0].text.strip()]
    for previous_index, current_index in zip(indices, indices[1:]):
        previous_bbox = local_bboxes[previous_index]
        current_bbox = local_bboxes[current_index]
        signed_gap = current_bbox[0] - previous_bbox[2]
        glyph_width = statistics.median(
            [
                width
                for member in (lines[previous_index], lines[current_index])
                if (width := _median_native_glyph_width(member, page_size)) is not None
            ]
            or [1.0]
        )
        separator = "" if signed_gap <= max(0.5, 0.25 * glyph_width) else " "
        content_parts.extend([separator, lines[current_index].text.strip()])

    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in members]),
        angle=members[0].angle,
        source_index=min(member.source_index for member in members),
        chars=[char for member in members for char in member.chars],
        visual_row_id=min(
            (member.visual_row_id for member in members if member.visual_row_id is not None),
            default=None,
        ),
        run_index=min(member.run_index for member in members),
        effective_height=statistics.median(member.effective_height for member in members),
        font_signature=members[0].font_signature,
        font_coverage=min(member.font_coverage for member in members),
        split_from_row=any(member.split_from_row for member in members),
    )
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _median_native_glyph_width(line: _LineItem, page_size: tuple[float, float]) -> float | None:
    """返回单个原生 run 的可见字符中位宽度，缺少字符时返回空。"""

    widths: list[float] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if not raw_char.isprintable() or raw_char.isspace() or bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        widths.append(max(0.1, local_bbox[2] - local_bbox[0]))
    return statistics.median(widths) if widths else None


def _build_formula_like_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[_LineItem]]:
    """仅依据栏带、右侧短锚点和空间连通关系聚合公式状区域。"""

    blocks: list[dict[str, Any]] = []
    claimed_source_indices: set[int] = set()
    for angle in sorted({line.angle for line in lines}):
        angle_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle
        ]
        if len(angle_geometry) < 2:
            continue
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in angle_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(angle_geometry, local_page_width, median_height)
        for lane in lanes:
            if lane.is_span or len(lane.lines) < 2:
                continue
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            anchors = _find_formula_spatial_anchors(lane, median_height)
            if not anchors:
                continue
            anchor_centers = [_bbox_center_y(anchor_bbox) for _anchor, anchor_bbox in anchors]
            for anchor_index, anchor_geometry in enumerate(anchors):
                anchor_line, anchor_bbox = anchor_geometry
                if anchor_line.source_index in claimed_source_indices:
                    continue
                anchor_height = _line_effective_height(anchor_line, anchor_bbox)
                band_top = _bbox_center_y(anchor_bbox) - 2.25 * anchor_height
                band_bottom = _bbox_center_y(anchor_bbox) + 2.25 * anchor_height
                if anchor_index > 0:
                    band_top = max(
                        band_top,
                        (anchor_centers[anchor_index - 1] + anchor_centers[anchor_index]) / 2.0,
                    )
                if anchor_index + 1 < len(anchors):
                    band_bottom = min(
                        band_bottom,
                        (anchor_centers[anchor_index] + anchor_centers[anchor_index + 1]) / 2.0,
                    )
                members = _grow_formula_spatial_component(
                    lane,
                    anchor_geometry,
                    band_top,
                    band_bottom,
                    claimed_source_indices,
                    table_bboxes,
                )
                if len(members) < 2:
                    continue
                if len(members) == 2 and _bbox_axis_overlap_ratio(
                    members[0][1],
                    members[1][1],
                    axis="y",
                ) < 0.2:
                    continue
                block = _formula_members_to_block(
                    members,
                    page_size,
                    angle,
                    anchor_source_index=anchor_line.source_index,
                )
                if block is None:
                    continue
                blocks.append(block)
                claimed_source_indices.update(line.source_index for line, _bbox in members)

    remaining_lines = [line for line in lines if line.source_index not in claimed_source_indices]
    return blocks, remaining_lines


def _find_formula_spatial_anchors(
    lane: _TextLane,
    median_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """查找栏带右缘的孤立短块，并要求同高度左侧存在可连接短行。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_interval = _formula_lane_body_interval(lane, median_height)
    if body_interval is None:
        return []
    body_top, body_bottom = body_interval
    anchors: list[tuple[_LineItem, BBox]] = []
    for line, bbox in lane.lines:
        line_height = _line_effective_height(line, bbox)
        line_width = bbox[2] - bbox[0]
        if line_width > max(4.0 * line_height, 0.12 * lane_width):
            continue
        same_row_fragments = [
            other_line
            for other_line, _other_bbox in lane.lines
            if line.visual_row_id is not None
            and other_line.visual_row_id == line.visual_row_id
            and (line.split_from_row or other_line.split_from_row)
        ]
        if len(same_row_fragments) >= 3:
            # 一条粗行被多个大空格拆成密集词组时更像普通排版行，不能把末词当作公式编号锚点。
            continue
        if abs(lane.right - bbox[2]) > max(3.0, 0.04 * lane_width):
            continue
        center_y = _bbox_center_y(bbox)
        if not body_top <= center_y <= body_bottom:
            continue
        has_left_peer = any(
            other_line.source_index != line.source_index
            and other_bbox[2] - other_bbox[0] <= 0.75 * lane_width
            and _bbox_center_x(other_bbox) < bbox[0]
            and _formula_seed_vertical_match(
                bbox,
                line_height,
                other_bbox,
                _line_effective_height(other_line, other_bbox),
            )
            for other_line, other_bbox in lane.lines
        )
        if has_left_peer:
            anchors.append((line, bbox))
    return _deduplicate_formula_anchors(anchors, median_height)


def _formula_lane_body_interval(
    lane: _TextLane,
    median_height: float,
) -> tuple[float, float] | None:
    """用连续出现的常规宽行确定栏带正文纵向范围，排除孤立页眉。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_lines = sorted(
        (
            item
            for item in lane.lines
            if item[1][2] - item[1][0]
            >= max(4.0 * _line_effective_height(*item), 0.35 * lane_width)
        ),
        key=lambda item: (item[1][1], item[1][0]),
    )
    if len(body_lines) < 3:
        return None
    dense_lines: list[tuple[_LineItem, BBox]] = []
    for index, item in enumerate(body_lines):
        has_close_previous = index > 0 and item[1][1] - body_lines[index - 1][1][3] <= 1.5 * median_height
        has_close_next = (
            index + 1 < len(body_lines)
            and body_lines[index + 1][1][1] - item[1][3] <= 1.5 * median_height
        )
        if has_close_previous or has_close_next:
            dense_lines.append(item)
    if len(dense_lines) < 3:
        return None
    return (
        min(bbox[1] for _line, bbox in dense_lines),
        max(bbox[3] for _line, bbox in dense_lines),
    )


def _deduplicate_formula_anchors(
    anchors: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """同一高度出现多个右缘短块时只保留最靠右的空间锚点。"""

    if not anchors:
        return []
    output: list[tuple[_LineItem, BBox]] = []
    tolerance = max(1.5, 0.35 * median_height)
    for anchor in sorted(anchors, key=lambda item: (_bbox_center_y(item[1]), -item[1][2])):
        if output and abs(_bbox_center_y(anchor[1]) - _bbox_center_y(output[-1][1])) <= tolerance:
            if (anchor[1][2], -anchor[1][0]) > (output[-1][1][2], -output[-1][1][0]):
                output[-1] = anchor
            continue
        output.append(anchor)
    return output


def _grow_formula_spatial_component(
    lane: _TextLane,
    anchor_geometry: tuple[_LineItem, BBox],
    band_top: float,
    band_bottom: float,
    claimed_source_indices: set[int],
    table_bboxes: list[BBox],
) -> list[tuple[_LineItem, BBox]]:
    """从右缘锚点的左侧首批成员出发，按二维邻接扩展公式分量。"""

    anchor_line, anchor_bbox = anchor_geometry
    lane_width = max(0.1, lane.right - lane.left)
    candidates = [
        item
        for item in lane.lines
        if item[0].source_index not in claimed_source_indices
        and item[1][2] - item[1][0] <= 0.75 * lane_width
        and band_top <= _bbox_center_y(item[1]) <= band_bottom
    ]
    seeds = [
        item
        for item in candidates
        if item[0].source_index != anchor_line.source_index
        and _bbox_center_x(item[1]) < anchor_bbox[0]
        and _formula_seed_vertical_match(
            anchor_bbox,
            _line_effective_height(anchor_line, anchor_bbox),
            item[1],
            _line_effective_height(*item),
        )
        and not _connection_crosses_table(anchor_line.bbox, item[0].bbox, table_bboxes)
    ]
    if not seeds:
        return []

    members = [anchor_geometry, *seeds]
    member_sources = {line.source_index for line, _bbox in members}
    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            candidate_line, candidate_bbox = candidate
            if candidate_line.source_index in member_sources:
                continue
            if any(
                _formula_lines_are_connected(
                    member_line,
                    member_bbox,
                    candidate_line,
                    candidate_bbox,
                    table_bboxes,
                )
                for member_line, member_bbox in members
            ):
                members.append(candidate)
                member_sources.add(candidate_line.source_index)
                changed = True
    return members


def _formula_seed_vertical_match(
    anchor_bbox: BBox,
    anchor_height: float,
    candidate_bbox: BBox,
    candidate_height: float,
) -> bool:
    """判断左侧短行是否与右缘锚点处在同一公式高度带。"""

    overlap_ratio = _bbox_axis_overlap_ratio(anchor_bbox, candidate_bbox, axis="y")
    center_difference = abs(_bbox_center_y(anchor_bbox) - _bbox_center_y(candidate_bbox))
    return overlap_ratio >= 0.3 or center_difference <= 0.6 * max(anchor_height, candidate_height)


def _formula_lines_are_connected(
    first_line: _LineItem,
    first_bbox: BBox,
    second_line: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """按垂直接近和水平覆盖判断两个公式成员是否空间连通。"""

    if first_line.angle != second_line.angle:
        return False
    if _connection_crosses_table(first_line.bbox, second_line.bbox, table_bboxes):
        return False
    first_height = _line_effective_height(first_line, first_bbox)
    second_height = _line_effective_height(second_line, second_bbox)
    pair_height = max(first_height, second_height)
    vertical_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="y")
    vertical_gap = max(first_bbox[1] - second_bbox[3], second_bbox[1] - first_bbox[3], 0.0)
    if vertical_overlap < 0.2 and vertical_gap > 0.6 * pair_height:
        return False
    horizontal_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x")
    horizontal_gap = max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)
    return horizontal_overlap > 0.0 or horizontal_gap <= 1.5 * pair_height


def _is_detached_formula_sidecar(
    anchor: tuple[_LineItem, BBox],
    members: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> bool:
    """仅依据 bbox 判断右侧锚点是否为与公式主体分离的窄幅 sidecar。"""

    anchor_line, anchor_bbox = anchor
    body_bboxes = [
        bbox
        for line, bbox in members
        if line.source_index != anchor_line.source_index
    ]
    if not body_bboxes:
        return False

    body_bbox = _bbox_union_many(body_bboxes)
    component_bbox = _bbox_union(body_bbox, anchor_bbox)
    effective_height = max(0.1, median_height)
    anchor_width = max(0.0, anchor_bbox[2] - anchor_bbox[0])
    component_width = max(0.1, component_bbox[2] - component_bbox[0])
    horizontal_gap = anchor_bbox[0] - body_bbox[2]
    right_tolerance = max(0.5, 0.1 * effective_height)
    minimum_gap = max(2.5 * effective_height, 0.08 * component_width)

    return (
        anchor_bbox[0] >= body_bbox[2]
        and anchor_bbox[2] >= component_bbox[2] - right_tolerance
        and anchor_width <= 2.0 * effective_height
        and horizontal_gap > minimum_gap
    )


def _formula_members_to_block(
    members: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
    angle: int,
    *,
    anchor_source_index: int,
) -> dict[str, Any] | None:
    """把公式空间分量按视觉行聚类，并将非末视觉行的离散右侧 sidecar 后置。"""

    heights = [_line_effective_height(line, bbox) for line, bbox in members]
    median_height = statistics.median(heights) if heights else 1.0
    row_tolerance = max(1.5, 0.35 * median_height)
    rows: list[list[tuple[_LineItem, BBox]]] = []
    for member in sorted(members, key=lambda item: (_bbox_center_y(item[1]), item[1][0], item[0].source_index)):
        if not rows:
            rows.append([member])
            continue
        row_center = statistics.median(_bbox_center_y(bbox) for _line, bbox in rows[-1])
        if abs(_bbox_center_y(member[1]) - row_center) <= row_tolerance:
            rows[-1].append(member)
        else:
            rows.append([member])

    trailing_sidecar_content: str | None = None
    # 右侧 sidecar 按视觉 y 常落在分式中部；仅在其后仍有公式行时转为逻辑末行。
    for row_index, row in enumerate(rows[:-1]):
        anchor_member = next(
            (member for member in row if member[0].source_index == anchor_source_index),
            None,
        )
        if anchor_member is None:
            continue
        if _is_detached_formula_sidecar(anchor_member, members, median_height):
            rows[row_index] = [
                member for member in row if member[0].source_index != anchor_source_index
            ]
            trailing_sidecar_content = anchor_member[0].text.strip()
        break

    row_contents = [_join_formula_visual_row(row, page_size) for row in rows if row]
    if trailing_sidecar_content is not None:
        row_contents.append(trailing_sidecar_content)
    content = _sanitize_pdf_control_text("\n".join(filter(None, row_contents)), preserve_newlines=True)
    if not content.strip():
        return None
    return {
        "type": "text",
        "bbox": _bbox_union_many([line.bbox for line, _bbox in members]),
        "angle": angle,
        "content": content,
    }


def _join_formula_visual_row(
    row: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
) -> str:
    """将一个公式视觉行按局部 x 排序，并按字宽估计几何空格。"""

    ordered = sorted(row, key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    if not ordered:
        return ""
    parts = [ordered[0][0].text.strip()]
    for previous, current in zip(ordered, ordered[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        gap = current_bbox[0] - previous_bbox[2]
        pair_height = max(
            _line_effective_height(previous_line, previous_bbox),
            _line_effective_height(current_line, current_bbox),
        )
        glyph_widths = [
            width
            for line in (previous_line, current_line)
            if (width := _median_native_glyph_width(line, page_size)) is not None
        ]
        glyph_width = statistics.median(glyph_widths) if glyph_widths else max(1.0, 0.5 * pair_height)
        if gap <= max(0.5, 0.2 * pair_height):
            separator = ""
        else:
            separator = " " * max(1, min(8, int(round(gap / glyph_width))))
        parts.extend([separator, current_line.text.strip()])
    return "".join(parts).strip()


def _restore_dense_split_visual_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在公式认领后恢复同一栏带内被均匀大空格拆开的密集原生视觉行。"""

    if len(lines) < 3:
        return list(lines)
    lane_keys: dict[int, tuple[int, int]] = {}
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle
        ]
        if not line_geometry:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane_index, lane in enumerate(lanes):
            if lane.is_span:
                continue
            for line, _bbox in lane.lines:
                lane_keys[line.source_index] = (angle, lane_index)

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    consumed_source_indices: set[int] = set()
    restored_lines: list[_LineItem] = []
    for members in row_groups.values():
        if not _can_restore_dense_split_visual_row(
            members,
            page_size,
            table_bboxes,
            lane_keys,
        ):
            continue
        restored_lines.append(_merge_dense_split_visual_row(members, page_size))
        consumed_source_indices.update(member.source_index for member in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(restored_lines)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _can_restore_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
    lane_keys: dict[int, tuple[int, int]],
) -> bool:
    """检查 hard-split run 是否构成同栏、同字体且占用充分的完整视觉行。"""

    if len(members) < 3 or not all(member.split_from_row for member in members):
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != list(range(len(ordered))):
        return False
    member_lane_keys = {lane_keys.get(member.source_index) for member in ordered}
    if None in member_lane_keys or len(member_lane_keys) != 1:
        return False
    font_signatures = {member.font_signature for member in ordered}
    if len(font_signatures) != 1:
        return False
    if any(
        _bbox_intersects(member.bbox, table_bbox)
        for member in ordered
        for table_bbox in table_bboxes
    ):
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
        )
        for member in ordered
    ]
    local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    heights = [_line_effective_height(member, bbox) for member, bbox in local_geometry]
    median_height = statistics.median(heights)
    glyph_widths = [
        width
        for member, _bbox in local_geometry
        if (width := _median_native_glyph_width(member, page_size)) is not None
    ]
    median_glyph_width = statistics.median(glyph_widths) if glyph_widths else 0.0
    gap_limit = max(12.0, 1.75 * median_height, 3.0 * median_glyph_width)
    for previous, current in zip(local_geometry, local_geometry[1:]):
        if not _same_baseline_geometry(
            previous[1],
            _line_effective_height(*previous),
            current[1],
            _line_effective_height(*current),
            maximum_gap=gap_limit,
        ):
            return False

    union_bbox = _bbox_union_many([bbox for _member, bbox in local_geometry])
    occupied_width = sum(bbox[2] - bbox[0] for _member, bbox in local_geometry)
    return occupied_width / max(0.1, union_bbox[2] - union_bbox[0]) >= 0.65


def _merge_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序恢复密集视觉行，正净空使用单空格，重叠片段直接连接。"""

    ordered_geometry = sorted(
        (
            (
                member,
                _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
            )
            for member in members
        ),
        key=lambda item: (item[1][0], item[1][1], item[0].source_index),
    )
    content_parts = [ordered_geometry[0][0].text.strip()]
    for previous, current in zip(ordered_geometry, ordered_geometry[1:]):
        separator = "" if current[1][0] <= previous[1][2] else " "
        content_parts.extend([separator, current[0].text.strip()])

    ordered_members = [member for member, _bbox in ordered_geometry]
    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in ordered_members]),
        angle=ordered_members[0].angle,
        source_index=min(member.source_index for member in ordered_members),
        chars=[char for member in ordered_members for char in member.chars],
        visual_row_id=ordered_members[0].visual_row_id,
        run_index=0,
        effective_height=statistics.median(
            _line_effective_height(member, bbox)
            for member, bbox in ordered_geometry
        ),
        font_signature=ordered_members[0].font_signature,
        font_coverage=min(member.font_coverage for member in ordered_members),
        split_from_row=False,
    )
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _build_text_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    drawing_lines: list[_AxisLine] | None = None,
) -> list[dict[str, Any]]:
    """将剩余文本行按局部栏带和自然段边界聚成统一 text block。"""

    blocks: list[dict[str, Any]] = []
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        local_axis_lines = _transform_axis_lines(drawing_lines or [], page_size, angle)

        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            if not lane.lines:
                continue
            regular_gap, gap_mad = _estimate_lane_gap(lane)
            component: list[tuple[_LineItem, BBox]] = [lane.lines[0]]
            components: list[list[tuple[_LineItem, BBox]]] = []
            for previous, current in zip(lane.lines, lane.lines[1:]):
                if _should_connect_text_rows(
                    previous,
                    current,
                    lane,
                    regular_gap,
                    gap_mad,
                    table_bboxes,
                    local_axis_lines,
                ):
                    component.append(current)
                else:
                    components.append(component)
                    component = [current]
            components.append(component)

            for component_geometry in components:
                component_lines = [item[0] for item in component_geometry]
                content = _merge_text_line_content([line.text for line in component_lines])
                if not content:
                    continue
                visual_row_ids = {
                    line.visual_row_id for line in component_lines if line.visual_row_id is not None
                }
                single_run_row_id = (
                    component_lines[0].visual_row_id
                    if len(component_lines) == 1
                    and component_lines[0].split_from_row
                    and component_lines[0].visual_row_id is not None
                    else None
                )
                blocks.append(
                    {
                        "type": "text",
                        "bbox": _bbox_union_many([line.bbox for line in component_lines]),
                        "angle": angle,
                        "content": content,
                        "_visual_row_ids": visual_row_ids,
                        "_single_run_row_id": single_run_row_id,
                    }
                )
    return blocks


def _line_effective_height(line: _LineItem, local_bbox: BBox) -> float:
    """返回字符统计得到的有效行高，缺失时回退到局部 bbox 高度。"""

    return max(0.1, line.effective_height or (local_bbox[3] - local_bbox[1]))


def _infer_text_lanes(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    median_height: float,
) -> list[_TextLane]:
    """从重复左右边缘推断稳定栏带，并把跨栏行放入独立 span lane。"""

    anchor_tolerance = max(3.0, 0.75 * median_height)
    regular_lines = [
        item
        for item in line_geometry
        if item[1][2] - item[1][0] >= max(4.0 * _line_effective_height(*item), 0.15 * local_page_width)
    ]
    left_clusters: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(regular_lines, key=lambda value: value[1][0]):
        if not left_clusters:
            left_clusters.append([item])
            continue
        cluster_left = statistics.median(member[1][0] for member in left_clusters[-1])
        if abs(item[1][0] - cluster_left) <= anchor_tolerance:
            left_clusters[-1].append(item)
        else:
            left_clusters.append([item])

    supported_intervals = [
        (
            statistics.median(item[1][0] for item in cluster),
            statistics.median(item[1][2] for item in cluster),
            len(cluster),
        )
        for cluster in left_clusters
        if len(cluster) >= 3
    ]
    supported_intervals.sort(key=lambda interval: interval[0])
    filtered_intervals: list[tuple[float, float, int]] = []
    for interval in supported_intervals:
        if not filtered_intervals:
            filtered_intervals.append(interval)
            continue
        previous = filtered_intervals[-1]
        minimum_gutter = max(12.0, 2.0 * median_height)
        if interval[0] - previous[1] >= minimum_gutter:
            filtered_intervals.append(interval)
        elif interval[2] > previous[2]:
            filtered_intervals[-1] = interval

    if not filtered_intervals:
        source = regular_lines or line_geometry
        filtered_intervals = [
            (
                min(item[1][0] for item in source),
                max(item[1][2] for item in source),
                len(source),
            )
        ]

    lanes = [_TextLane(left=left, right=right) for left, right, _support in filtered_intervals]
    span_lines: list[tuple[_LineItem, BBox]] = []
    for item in line_geometry:
        bbox = item[1]
        line_width = max(0.1, bbox[2] - bbox[0])
        scored_lanes = [
            (
                max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left)) / line_width,
                lane,
            )
            for lane in lanes
        ]
        best_coverage, best_lane = max(scored_lanes, key=lambda value: value[0])
        if len(lanes) == 1 or (
            best_coverage >= 0.5
            and best_lane.left - anchor_tolerance
            <= _bbox_center_x(bbox)
            <= best_lane.right + anchor_tolerance
        ):
            best_lane.lines.append(item)
        else:
            span_lines.append(item)

    if span_lines:
        lanes.append(
            _TextLane(
                left=min(item[1][0] for item in span_lines),
                right=max(item[1][2] for item in span_lines),
                lines=span_lines,
                is_span=True,
            )
        )
    return lanes


def _estimate_lane_gap(lane: _TextLane) -> tuple[float, float]:
    """从栏带内兼容相邻行的较小间隙簇估计常规净空和 MAD。"""

    lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    heights = [_line_effective_height(line, bbox) for line, bbox in lane.lines]
    median_height = statistics.median(heights) if heights else 1.0
    gaps: list[float] = []
    for previous, current in zip(lane.lines, lane.lines[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        if previous_line.visual_row_id == current_line.visual_row_id and (
            previous_line.split_from_row or current_line.split_from_row
        ):
            continue
        previous_height = _line_effective_height(previous_line, previous_bbox)
        current_height = _line_effective_height(current_line, current_bbox)
        if max(previous_height, current_height) / min(previous_height, current_height) > 1.35:
            continue
        gap = current_bbox[1] - previous_bbox[3]
        if gap < 0 or gap > 2.0 * max(previous_height, current_height):
            continue
        if _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5 and abs(
            previous_bbox[0] - current_bbox[0]
        ) > 1.5 * median_height:
            continue
        gaps.append(gap)

    if not gaps:
        return 0.35 * median_height, 0.0
    sorted_gaps = sorted(gaps)
    lower_count = max(1, math.ceil(len(sorted_gaps) * 0.6))
    lower_gaps = sorted_gaps[:lower_count]
    regular_gap = statistics.median(lower_gaps)
    gap_mad = statistics.median(abs(gap - regular_gap) for gap in lower_gaps)
    return regular_gap, gap_mad


def _should_connect_text_rows(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """综合局部间距、首行缩进、字体和障碍判断两个相邻视觉行是否同段。"""

    previous_line, previous_bbox = previous
    current_line, current_bbox = current
    previous_height = _line_effective_height(previous_line, previous_bbox)
    current_height = _line_effective_height(current_line, current_bbox)
    pair_height = max(previous_height, current_height)
    if previous_line.visual_row_id == current_line.visual_row_id and (
        previous_line.split_from_row or current_line.split_from_row
    ):
        return False
    if max(previous_height, current_height) / min(previous_height, current_height) > 1.35:
        return False

    vertical_gap = current_bbox[1] - previous_bbox[3]
    if vertical_gap < -0.25 * pair_height:
        return False
    if _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5 and abs(
        previous_bbox[0] - current_bbox[0]
    ) > 1.5 * pair_height:
        return False
    if _connection_crosses_table(previous_line.bbox, current_line.bbox, table_bboxes):
        return False
    if _horizontal_rule_separates_rows(previous_bbox, current_bbox, lane, axis_lines):
        return False

    gap_limit = max(
        regular_gap + max(0.5 * pair_height, 3.0 * gap_mad),
        1.1 * pair_height,
    )
    # 排版断词可以跳过缩进、字体和短行规则，但仍须限制在邻近物理行内，
    # 避免页内远距离的 “cross-” 与后续标题被误拼为同一段。
    if is_hyphen_at_line_end(previous_line.text):
        return vertical_gap <= max(gap_limit, 1.8 * pair_height)
    if vertical_gap > gap_limit:
        return False

    lane_width = max(0.1, lane.right - lane.left)
    next_indent = current_bbox[0] - lane.left
    previous_fill = max(0.0, previous_bbox[2] - lane.left) / lane_width
    terminal_previous = bool(re.search(r"[.!?。！？:：;；][\]\)}）】》”’'\"]*$", previous_line.text.rstrip()))
    if next_indent >= max(5.0, 0.65 * pair_height) and (previous_fill <= 0.8 or terminal_previous):
        return False

    previous_width = previous_bbox[2] - previous_bbox[0]
    current_width = current_bbox[2] - current_bbox[0]
    abnormal_gap = vertical_gap > regular_gap + 0.25 * pair_height
    if (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature != current_line.font_signature
        and (abnormal_gap or min(previous_width, current_width) <= 0.7 * lane_width)
    ):
        return False
    if abnormal_gap and min(previous_width, current_width) <= 0.65 * lane_width:
        return False
    return True


def _horizontal_rule_separates_rows(
    previous_bbox: BBox,
    current_bbox: BBox,
    lane: _TextLane,
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """检查两个相邻文本行之间是否存在覆盖当前栏带的长水平规则线。"""

    if current_bbox[1] <= previous_bbox[3]:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    for axis_line in axis_lines:
        if axis_line.orientation != "horizontal":
            continue
        line_y = _bbox_center_y(axis_line.bbox)
        if not previous_bbox[3] <= line_y <= current_bbox[1]:
            continue
        overlap = max(0.0, min(axis_line.bbox[2], lane.right) - max(axis_line.bbox[0], lane.left))
        if overlap / lane_width >= 0.6:
            return True
    return False


def _merge_text_line_content(line_texts: Sequence[str]) -> str:
    """按 Hybrid 语言与行末连字规则折叠普通文本行。"""

    normalized_lines = [_normalize_native_run_text(str(text or "")) for text in line_texts]
    normalized_lines = [text for text in normalized_lines if text]
    if not normalized_lines:
        return ""
    try:
        block_language = detect_lang("".join(normalized_lines))
    except Exception:
        block_language = ""
    content_parts = [normalized_lines[0]]
    for current_line in normalized_lines[1:]:
        processed_previous, separator = resolve_text_line_boundary(
            content_parts[-1],
            block_language=block_language,
            next_starts_with_lowercase=current_line[0].islower(),
        )
        content_parts[-1] = processed_previous
        content_parts.extend([separator, current_line])
    return "".join(content_parts).strip()


def _connection_crosses_table(
    first_bbox: BBox,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """检查两行中心连接区域是否穿过已确认表格。"""

    first_center = (_bbox_center_x(first_bbox), _bbox_center_y(first_bbox))
    second_center = (_bbox_center_x(second_bbox), _bbox_center_y(second_bbox))
    connector = _coerce_bbox(
        (
            min(first_center[0], second_center[0]) - 0.1,
            min(first_center[1], second_center[1]) - 0.1,
            max(first_center[0], second_center[0]) + 0.1,
            max(first_center[1], second_center[1]) + 0.1,
        )
    )
    return connector is not None and any(_bbox_intersects(connector, table_bbox) for table_bbox in table_bboxes)


def _sort_blocks_with_visual_row_groups(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把同一粗行拆出的单行 block 包装成虚拟项排序，再按局部 x 顺序展开。"""

    grouped_indices: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        row_id = block.get("_single_run_row_id")
        if isinstance(row_id, int):
            grouped_indices.setdefault(row_id, []).append(index)

    virtual_groups: dict[int, dict[str, Any]] = {}
    consumed_indices: set[int] = set()
    for row_id, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        members = [blocks[index] for index in indices]
        virtual_group = {
            "type": "_xycut_visual_row_group",
            "bbox": _bbox_union_many([member["bbox"] for member in members]),
            "angle": members[0].get("angle", 0),
            "content": "",
            "_members": members,
        }
        virtual_groups[row_id] = virtual_group
        consumed_indices.update(indices)

    sortable_blocks = [block for index, block in enumerate(blocks) if index not in consumed_indices]
    sortable_blocks.extend(virtual_groups.values())
    sorted_payloads = sort_entries(sortable_blocks)
    output: list[dict[str, Any]] = []
    for payload in sorted_payloads:
        members = payload.get("_members")
        if not isinstance(members, list):
            output.append(payload)
            continue
        angle = int(payload.get("angle", 0) or 0) % 360
        members.sort(
            key=lambda member: (
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[0],
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[1],
            )
        )
        output.extend(members)
    return output


def _normalize_output_block(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> dict[str, Any] | None:
    """在排序完成后将绝对 bbox 裁剪并归一化为 model_list 坐标。"""

    page_width, page_height = page_size
    bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page_size)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None
    content = block.get("content")
    if not isinstance(content, str):
        return None
    content = _sanitize_pdf_control_text(content, preserve_newlines=True)
    if not content.strip():
        return None
    normalized_bbox = _normalize_bbox_to_thousandths(bbox, page_size)
    return {
        "type": "table" if block.get("type") == "table" else "text",
        "bbox": normalized_bbox,
        "angle": int(block.get("angle", 0) or 0) % 360,
        "content": content,
    }


def _normalize_bbox_to_thousandths(
    bbox: BBox,
    page_size: tuple[float, float],
) -> list[float]:
    """将绝对 bbox 转成千分位刻度，并保证舍入后的宽高至少各占一个刻度。"""

    page_width, page_height = page_size
    ticks = [
        max(0, min(1000, int(round(bbox[0] / page_width * 1000)))),
        max(0, min(1000, int(round(bbox[1] / page_height * 1000)))),
        max(0, min(1000, int(round(bbox[2] / page_width * 1000)))),
        max(0, min(1000, int(round(bbox[3] / page_height * 1000)))),
    ]
    if ticks[2] <= ticks[0]:
        if ticks[0] < 1000:
            ticks[2] = ticks[0] + 1
        else:
            ticks[0] = max(0, ticks[2] - 1)
    if ticks[3] <= ticks[1]:
        if ticks[1] < 1000:
            ticks[3] = ticks[1] + 1
        else:
            ticks[1] = max(0, ticks[3] - 1)
    return [tick / 1000 for tick in ticks]


def _rotate_bbox_to_upright(
    bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
) -> BBox:
    """将页面 bbox 转到当前文本方向的正向局部坐标。"""

    page_width, page_height = page_size
    x0, y0, x1, y1 = bbox
    if angle == 270:
        return (page_height - y1, x0, page_height - y0, x1)
    if angle == 90:
        return (y0, page_width - x1, y1, page_width - x0)
    if angle == 180:
        return (page_width - x1, page_height - y1, page_width - x0, page_height - y0)
    return bbox


def _rotate_bbox_from_upright(
    bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
) -> BBox:
    """将正向局部 bbox 逆变换回 PDF 页面坐标。"""

    page_width, page_height = page_size
    x0, y0, x1, y1 = bbox
    if angle == 270:
        return (y0, page_height - x1, y1, page_height - x0)
    if angle == 90:
        return (page_width - y1, x0, page_width - y0, x1)
    if angle == 180:
        return (page_width - x1, page_height - y1, page_width - x0, page_height - y0)
    return bbox


def _axis_lines_intersect(
    horizontal: _LocalAxisLine,
    vertical: _LocalAxisLine,
    tolerance: float,
) -> bool:
    """检查一条横线与一条竖线是否在容差内相交。"""

    horizontal_y = _bbox_center_y(horizontal.bbox)
    vertical_x = _bbox_center_x(vertical.bbox)
    return (
        horizontal.bbox[0] - tolerance <= vertical_x <= horizontal.bbox[2] + tolerance
        and vertical.bbox[1] - tolerance <= horizontal_y <= vertical.bbox[3] + tolerance
    )


def _median_fragment_height(fragments: list[_Fragment]) -> float:
    """返回正向文本片段高度的中位数。"""

    heights = [
        fragment.local_bbox[3] - fragment.local_bbox[1]
        for fragment in fragments
        if fragment.local_bbox[3] > fragment.local_bbox[1]
    ]
    return max(0.1, float(statistics.median(heights)) if heights else 1.0)


def _coerce_bbox(value: Any) -> BBox | None:
    """将任意四元 bbox 规范成非退化浮点坐标。"""

    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _clip_bbox(
    bbox: BBox | None,
    page_size: tuple[float, float],
) -> BBox | None:
    """将 bbox 裁剪到页面范围，退化框返回 None。"""

    if bbox is None:
        return None
    page_width, page_height = page_size
    return _coerce_bbox(
        (
            max(0.0, min(page_width, bbox[0])),
            max(0.0, min(page_height, bbox[1])),
            max(0.0, min(page_width, bbox[2])),
            max(0.0, min(page_height, bbox[3])),
        )
    )


def _bbox_union(first: BBox, second: BBox) -> BBox:
    """返回两个 bbox 的外接并集框。"""

    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _bbox_union_many(bboxes: Sequence[BBox]) -> BBox:
    """返回非空 bbox 序列的外接并集框。"""

    if not bboxes:
        raise ValueError("bbox sequence must not be empty")
    result = bboxes[0]
    for bbox in bboxes[1:]:
        result = _bbox_union(result, bbox)
    return result


def _expand_bbox(bbox: BBox, margin: float) -> BBox:
    """向四周扩展 bbox，仅供几何容差判定使用。"""

    return (
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin,
    )


def _bbox_center_x(bbox: BBox) -> float:
    """返回 bbox 的水平中心。"""

    return (bbox[0] + bbox[2]) / 2.0


def _bbox_center_y(bbox: BBox) -> float:
    """返回 bbox 的垂直中心。"""

    return (bbox[1] + bbox[3]) / 2.0


def _bbox_intersects(first: BBox, second: BBox) -> bool:
    """检查两个 bbox 是否存在正面积交叠。"""

    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])


def _bbox_overlap_in_smaller(first: BBox, second: BBox) -> float:
    """计算交集面积占较小 bbox 面积的比例。"""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _bbox_axis_overlap_ratio(
    first: BBox,
    second: BBox,
    *,
    axis: Literal["x", "y"],
) -> float:
    """计算指定轴上的交叠长度占较短轴长的比例。"""

    if axis == "x":
        first_start, first_end = first[0], first[2]
        second_start, second_end = second[0], second[2]
    else:
        first_start, first_end = first[1], first[3]
        second_start, second_end = second[1], second[3]
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    shorter = min(first_end - first_start, second_end - second_start)
    return overlap / shorter if shorter > 0 else 0.0


def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    """检查点是否位于 bbox 内部或边界上。"""

    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def doc_analyze(
    pdf_bytes: bytes,
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    page_index_map: list[int] | None = None,
) -> list[list[dict[str, Any]]]:
    """使用 Flash 提取原生文本，OCR 文档则委托 Hybrid low。"""

    if parse_mode not in {"auto", "txt", "ocr"}:
        raise ValueError(f"parse_mode {parse_mode} is not supported")

    with PDFDocument(pdf_bytes) as pdf_doc:
        page_count = pdf_doc.page_count
        if page_index_map and len(page_index_map) != page_count:
            raise ValueError(
                f"Flash page_index_map length mismatch: page_count={page_count}, page_index_map={len(page_index_map)}"
            )

        resolved_mode: Literal["txt", "ocr"]
        if parse_mode == "auto":
            resolved_mode = pdf_doc.classify()
        else:
            resolved_mode = parse_mode

        if resolved_mode == "txt":
            return _analyze_native_document(pdf_doc)

    # OCR 路径延迟加载 Hybrid，保证原生 Flash 不引入本地视觉模型运行时。
    from mineru.backend.hybrid.analyze import doc_analyze as hybrid_doc_analyze

    _middle_json, model_list = hybrid_doc_analyze(
        pdf_bytes,
        effort="low",
        parse_mode="ocr",
        page_index_map=page_index_map,
    )
    return model_list


if __name__ == '__main__':
    # pdf_path = "/Users/myhloli/pdf/截断合并/demo1-3.pdf"
    # pdf_path = "/Users/myhloli/pdf/png/seal4.png"  # shubiao.png
    pdf_path = "/Users/myhloli/pdf/demo1.pdf"
    pdf_bytes = read_fn(pdf_path)
    model_list = doc_analyze(pdf_bytes, parse_mode="auto")
    logger.info(f"model_list: {model_list}")
