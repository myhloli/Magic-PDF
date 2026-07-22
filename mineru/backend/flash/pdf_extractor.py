"""Flash PDF 文本与表格提取器。

Flash 仅使用 PDF 原生文本或 OCR 文本行，不运行布局、公式、图片或表格结构模型。
"""

from __future__ import annotations

import math
import re
import statistics
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from loguru import logger
from pdftext.schema import Char

from mineru.backend.hybrid.table_text import project_ocr_table_text, project_pdf_table_text
from mineru.backend.utils.char_utils import resolve_text_line_boundary
from mineru.backend.utils.xycut_pp_sorter import sort_entries
from mineru.types import BBox
from mineru.utils.config_reader import get_processing_window_size
from mineru.utils.language import detect_lang
from mineru.utils.pdf_document import PDFDocument, get_lines_from_chars


_OCR_MIN_CONFIDENCE = 0.5
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


@dataclass(slots=True)
class _LineItem:
    """保存单个可视文本行及其原始几何信息。"""

    text: str
    bbox: BBox
    angle: int
    source_index: int
    score: float = 1.0
    chars: list[Char] = field(default_factory=list)
    pixel_quad: tuple[tuple[float, float], ...] | None = None


@dataclass(slots=True)
class _Fragment:
    """保存表格规则使用的单元文本片段。"""

    text: str
    bbox: BBox
    local_bbox: BBox
    line_index: int
    score: float = 1.0


@dataclass(slots=True)
class _VisualRow:
    """保存同一局部水平带内的表格片段。"""

    fragments: list[_Fragment]
    center_y: float
    bbox: BBox


@dataclass(slots=True)
class _AxisLine:
    """保存 PDF 路径或 OCR 图像中的横竖线。"""

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
class _PageSource:
    """保存单页分析所需的文本、字符和可选 OCR 图像。"""

    page_size: tuple[float, float]
    lines: list[_LineItem]
    chars: list[Char]
    drawing_lines: list[_AxisLine]
    render_scale: float = 1.0
    bgr_image: np.ndarray | None = None
    ocr_model: Any = None


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


def doc_analyze(
    pdf_bytes: bytes,
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    page_index_map: list[int] | None = None,
) -> list[list[dict[str, Any]]]:
    """使用 Flash 规则提取文本与空间投影表格，返回物理页顺序的 model_list。"""

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
        return _analyze_ocr_document(pdf_doc)


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
        model_list.append(_analyze_page_source(source, "txt"))
    return model_list


def _analyze_ocr_document(pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
    """按窗口渲染并 OCR 扫描 PDF，避免长文档同时常驻所有页图像。"""

    page_count = pdf_doc.page_count
    if page_count == 0:
        return []
    ocr_model, run_ocr_inference, sorted_boxes, crop_text_line = _load_ocr_runtime()
    window_size = get_processing_window_size(default=64)
    model_list: list[list[dict[str, Any]]] = []

    for start in range(0, page_count, window_size):
        end = min(page_count, start + window_size)
        page_images = pdf_doc.render_pages(start, end - 1)
        if len(page_images) != end - start:
            for page_image in page_images:
                page_image.pil_image.close()
            raise ValueError(f"Flash OCR rendered page count mismatch: expected={end - start}, actual={len(page_images)}")

        try:
            bgr_images = [_pil_image_to_bgr(page_image.pil_image) for page_image in page_images]
            window_lines = _run_full_page_ocr(
                ocr_model,
                bgr_images,
                [page_image.scale for page_image in page_images],
                [pdf_doc.page_size(page_idx) for page_idx in range(start, end)],
                run_ocr_inference,
                sorted_boxes,
                crop_text_line,
            )
            for offset, (page_image, bgr_image, lines) in enumerate(zip(page_images, bgr_images, window_lines)):
                page_size = pdf_doc.page_size(start + offset)
                median_height = _median_line_height(lines)
                drawing_lines = _collect_ocr_drawing_lines(
                    pdf_doc,
                    start + offset,
                    bgr_image,
                    page_image.scale,
                    median_height,
                )
                source = _PageSource(
                    page_size=page_size,
                    lines=lines,
                    chars=[],
                    drawing_lines=drawing_lines,
                    render_scale=page_image.scale,
                    bgr_image=bgr_image,
                    ocr_model=(ocr_model, run_ocr_inference),
                )
                model_list.append(_analyze_page_source(source, "ocr"))
        finally:
            for page_image in page_images:
                page_image.pil_image.close()

    return model_list


def _pil_image_to_bgr(pil_image: Any) -> np.ndarray:
    """将 PIL 页图像转为 OCR 使用的 BGR 数组，并及时关闭临时 RGB 副本。"""

    rgb_image = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
    try:
        return np.asarray(rgb_image)[:, :, ::-1].copy()
    finally:
        if rgb_image is not pil_image:
            rgb_image.close()


def _load_ocr_runtime() -> tuple[Any, Any, Any, Any]:
    """延迟加载 OCR 依赖，保证 txt 路径不导入本地视觉模型运行时。"""

    try:
        from mineru.backend.local_model_runtime import (
            AtomicModel,
            AtomModelSingleton,
            run_ocr_inference,
        )
        from mineru.utils.ocr_utils import (
            get_rotate_crop_image_for_text_rec,
            sorted_boxes,
        )

        ocr_model = AtomModelSingleton().get_atom_model(
            AtomicModel.OCR,
            det_db_box_thresh=0.5,
            lang="ch",
            det_db_unclip_ratio=1.6,
            enable_merge_det_boxes=False,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Flash OCR dependencies are unavailable. Install the OCR runtime with `pip install 'mineru[medium]'`."
        ) from exc
    return ocr_model, run_ocr_inference, sorted_boxes, get_rotate_crop_image_for_text_rec


def _run_full_page_ocr(
    ocr_model: Any,
    bgr_images: list[np.ndarray],
    render_scales: list[float],
    page_sizes: list[tuple[float, float]],
    run_ocr_inference: Any,
    sorted_boxes_fn: Any,
    crop_text_line: Any,
) -> list[list[_LineItem]]:
    """先批量检测全页文本框，再将所有裁图合并执行一次 OCR rec。"""

    if not bgr_images:
        return []
    det_results = run_ocr_inference(
        ocr_model.text_detector.batch_predict,
        bgr_images,
        max_batch_size=min(16, len(bgr_images)),
        tqdm_enable=False,
    )
    if len(det_results) != len(bgr_images):
        raise ValueError(f"Flash OCR det result count mismatch: images={len(bgr_images)}, results={len(det_results)}")

    page_boxes: list[list[np.ndarray]] = []
    crop_images: list[np.ndarray] = []
    crop_mapping: list[tuple[int, int]] = []
    for page_offset, (bgr_image, det_result) in enumerate(zip(bgr_images, det_results)):
        raw_boxes = det_result[0] if det_result else None
        sorted_page_boxes = list(sorted_boxes_fn(raw_boxes)) if raw_boxes is not None else []
        valid_boxes: list[np.ndarray] = []
        for raw_box in sorted_page_boxes:
            try:
                box = np.asarray(raw_box, dtype=np.float32).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if box.shape[0] < 4 or not np.isfinite(box).all():
                continue
            crop = crop_text_line(bgr_image, box)
            if crop is None or crop.size == 0:
                continue
            valid_boxes.append(box)
            crop_images.append(crop)
            crop_mapping.append((page_offset, len(valid_boxes) - 1))
        page_boxes.append(valid_boxes)

    page_rec_results: list[list[tuple[str, float] | None]] = [[None] * len(boxes) for boxes in page_boxes]
    if crop_images:
        raw_rec_results = run_ocr_inference(
            ocr_model.ocr,
            crop_images,
            det=False,
            rec=True,
            tqdm_enable=False,
        )
        rec_results = raw_rec_results[0] if raw_rec_results else []
        if len(rec_results) != len(crop_images):
            raise ValueError(f"Flash OCR rec result count mismatch: crops={len(crop_images)}, results={len(rec_results)}")
        for mapping, rec_result in zip(crop_mapping, rec_results):
            try:
                text, score = rec_result
                normalized_result = (str(text or ""), float(score or 0.0))
            except (TypeError, ValueError):
                normalized_result = ("", 0.0)
            page_rec_results[mapping[0]][mapping[1]] = normalized_result

    output: list[list[_LineItem]] = []
    for boxes, rec_results, scale, page_size in zip(
        page_boxes,
        page_rec_results,
        render_scales,
        page_sizes,
    ):
        page_lines: list[_LineItem] = []
        for source_index, (box, rec_result) in enumerate(zip(boxes, rec_results)):
            if rec_result is None:
                continue
            text, score = rec_result
            text = text.strip()
            if not text or score < _OCR_MIN_CONFIDENCE:
                continue
            bbox = _quad_to_page_bbox(box, scale, page_size)
            if bbox is None:
                continue
            page_lines.append(
                _LineItem(
                    text=text,
                    bbox=bbox,
                    angle=_infer_ocr_angle(box),
                    source_index=source_index,
                    score=score,
                    pixel_quad=tuple((float(point[0]), float(point[1])) for point in box),
                )
            )
        _promote_vertical_ocr_line_angles(page_lines)
        output.append(page_lines)
    return output


def _quad_to_page_bbox(
    quad: np.ndarray,
    scale: float,
    page_size: tuple[float, float],
) -> BBox | None:
    """将 OCR 像素四点框转为裁剪后的 PDF point bbox。"""

    if scale <= 0:
        return None
    page_width, page_height = page_size
    x0 = max(0.0, min(page_width, float(np.min(quad[:, 0])) / scale))
    y0 = max(0.0, min(page_height, float(np.min(quad[:, 1])) / scale))
    x1 = max(0.0, min(page_width, float(np.max(quad[:, 0])) / scale))
    y1 = max(0.0, min(page_height, float(np.max(quad[:, 1])) / scale))
    return _coerce_bbox((x0, y0, x1, y1))


def _infer_ocr_angle(quad: np.ndarray) -> int:
    """仅根据四点框首边方向返回可靠的标准 OCR 角度。"""

    points = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4:
        return 0
    edge_angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
    normalized = int(round(edge_angle / 90.0) * 90) % 360
    return normalized if normalized in {0, 90, 180, 270} else 0


def _promote_vertical_ocr_line_angles(lines: list[_LineItem]) -> None:
    """仅在多个窄高多字符框形成主导证据时，将 OCR 行提升为竖排方向。"""

    eligible: list[tuple[_LineItem, float, float]] = []
    for line in lines:
        if line.angle not in {0, 180} or line.pixel_quad is None:
            continue
        effective_text = "".join(char for char in line.text if not char.isspace())
        if len(effective_text) < 2:
            continue
        points = np.asarray(line.pixel_quad, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 4:
            continue
        top_width = float(np.linalg.norm(points[1] - points[0]))
        side_height = float(np.linalg.norm(points[3] - points[0]))
        eligible.append((line, top_width, side_height))
    vertical_lines = [line for line, top_width, side_height in eligible if side_height >= max(1.0, top_width * 1.5)]
    if len(vertical_lines) < 3 or len(vertical_lines) / max(1, len(eligible)) < 0.6:
        return
    for line in vertical_lines:
        line.angle = 90


def _build_native_line_items(
    pdf_lines: Sequence[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    page_rotation: int = 0,
) -> list[_LineItem]:
    """将 pdftext line 规范成 Flash 内部文本行。"""

    items: list[_LineItem] = []
    for source_index, pdf_line in enumerate(pdf_lines):
        bbox = _clip_bbox(_coerce_bbox(pdf_line.get("bbox")), page_size)
        if bbox is None:
            continue
        spans = pdf_line.get("spans") or []
        text = "".join(str(span.get("text") or "") for span in spans)
        text = text.replace("\r", "").replace("\n", "").strip()
        if not text:
            continue
        chars = [char for span in spans for char in (span.get("chars") or []) if isinstance(char, dict)]
        items.append(
            _LineItem(
                text=text,
                bbox=bbox,
                angle=(_normalize_pdftext_angle(pdf_line.get("rotation")) + int(page_rotation or 0)) % 360,
                source_index=source_index,
                chars=chars,
            )
        )
    return items


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


def _extract_image_drawing_lines(
    bgr_image: np.ndarray,
    render_scale: float,
    median_line_height: float,
) -> list[_AxisLine]:
    """从 OCR 已渲染图像中提取足够长且足够细的横竖线。"""

    if bgr_image.size == 0 or render_scale <= 0:
        return []
    try:
        import cv2
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Flash OCR image-line dependencies are unavailable. Install the OCR runtime with `pip install 'mineru[medium]'`."
        ) from exc

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    image_height, image_width = gray.shape[:2]
    median_height_px = max(1.0, median_line_height * render_scale)
    max_thickness = max(5.0, median_height_px * 0.5)
    min_length = max(20.0 * render_scale, median_height_px * 4.0)
    kernels = {
        "horizontal": cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(15, int(round(image_width * 0.02))), 1),
        ),
        "vertical": cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(15, int(round(image_height * 0.02)))),
        ),
    }

    output: list[_AxisLine] = []
    for orientation, kernel in kernels.items():
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            length = width if orientation == "horizontal" else height
            thickness = height if orientation == "horizontal" else width
            if length < min_length or thickness > max_thickness:
                continue
            bbox = _coerce_bbox(
                (
                    x / render_scale,
                    y / render_scale,
                    (x + width) / render_scale,
                    (y + height) / render_scale,
                )
            )
            if bbox is None:
                continue
            output.append(
                _AxisLine(
                    bbox=bbox,
                    width=max(0.1, thickness / render_scale),
                    orientation=orientation,  # type: ignore[arg-type]
                )
            )
    return _merge_axis_lines(output, tolerance=2.0)


def _collect_ocr_drawing_lines(
    pdf_doc: PDFDocument,
    page_idx: int,
    bgr_image: np.ndarray,
    render_scale: float,
    median_line_height: float,
) -> list[_AxisLine]:
    """合并 OCR 页面的 PDF 矢量线与图像形态学线，并统一去重。"""

    vector_lines = _get_pdf_drawing_lines(pdf_doc, page_idx)
    raster_lines = _extract_image_drawing_lines(
        bgr_image,
        render_scale,
        median_line_height,
    )
    return _merge_axis_lines([*vector_lines, *raster_lines], tolerance=2.0)


def _analyze_page_source(
    source: _PageSource,
    parse_mode: Literal["txt", "ocr"],
) -> list[dict[str, Any]]:
    """在单页内先确认表格，再聚合剩余文本并排序。"""

    if not source.lines:
        return []
    candidates = _detect_table_candidates(source, parse_mode)
    table_blocks, claimed_line_indices = _materialize_table_blocks(
        source,
        candidates,
        parse_mode,
    )
    text_blocks = _build_text_blocks(
        [line for line in source.lines if line.source_index not in claimed_line_indices],
        [block["bbox"] for block in table_blocks],
        source.page_size,
    )
    absolute_blocks = table_blocks + text_blocks
    sorted_blocks = sort_entries(absolute_blocks)
    return [
        normalized for block in sorted_blocks if (normalized := _normalize_output_block(block, source.page_size)) is not None
    ]


def _detect_table_candidates(
    source: _PageSource,
    parse_mode: Literal["txt", "ocr"],
) -> list[_TableCandidate]:
    """融合多列文本稳定性与横竖线证据，生成高精度表格候选。"""

    candidates: list[_TableCandidate] = []
    angles = sorted({line.angle for line in source.lines})
    for angle in angles:
        angle_lines = [line for line in source.lines if line.angle == angle]
        if not angle_lines:
            continue
        fragments = _build_fragments(angle_lines, source.page_size, parse_mode)
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
    parse_mode: Literal["txt", "ocr"],
) -> list[_Fragment]:
    """将原生行拆成表格单元候选，OCR 行则直接作为单个片段。"""

    fragments: list[_Fragment] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        if parse_mode == "ocr" or not line.chars:
            fragments.append(
                _Fragment(
                    text=line.text,
                    bbox=line.bbox,
                    local_bbox=local_bbox,
                    line_index=line.source_index,
                    score=line.score,
                )
            )
            continue
        line_fragments = _split_native_line_fragments(line, page_size)
        if line_fragments:
            fragments.extend(line_fragments)
        else:
            fragments.append(
                _Fragment(
                    text=line.text,
                    bbox=line.bbox,
                    local_bbox=local_bbox,
                    line_index=line.source_index,
                    score=line.score,
                )
            )
    return fragments


def _split_native_line_fragments(
    line: _LineItem,
    page_size: tuple[float, float],
) -> list[_Fragment]:
    """根据绝对间隙和字符宽度比例拆分原生表格行。"""

    chars: list[tuple[str, BBox, BBox]] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if raw_char in {"\r", "\n"}:
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if bbox is None:
            continue
        chars.append(
            (
                raw_char,
                bbox,
                _rotate_bbox_to_upright(bbox, page_size, line.angle),
            )
        )
    chars.sort(key=lambda item: (item[2][0], item[2][1]))

    output: list[_Fragment] = []
    text_parts: list[str] = []
    original_bbox: BBox | None = None
    local_bbox: BBox | None = None
    previous_bbox: BBox | None = None
    widths: list[float] = []
    pending_space = False

    def flush_fragment() -> None:
        """提交当前文本片段，并清空字符累计状态。"""

        nonlocal text_parts, original_bbox, local_bbox, previous_bbox, widths, pending_space
        text = "".join(text_parts).strip()
        if text and original_bbox is not None and local_bbox is not None:
            output.append(
                _Fragment(
                    text=text,
                    bbox=original_bbox,
                    local_bbox=local_bbox,
                    line_index=line.source_index,
                    score=line.score,
                )
            )
        text_parts = []
        original_bbox = None
        local_bbox = None
        previous_bbox = None
        widths = []
        pending_space = False

    for raw_char, char_bbox, char_local_bbox in chars:
        if raw_char.isspace():
            if text_parts:
                pending_space = True
            continue
        if previous_bbox is not None and widths:
            gap = char_local_bbox[0] - previous_bbox[2]
            average_width = sum(widths) / len(widths)
            if gap >= 15.0 or (pending_space and gap > average_width * 2.2):
                flush_fragment()
            elif pending_space:
                text_parts.append(" ")
                pending_space = False
        text_parts.append(raw_char)
        original_bbox = char_bbox if original_bbox is None else _bbox_union(original_bbox, char_bbox)
        local_bbox = char_local_bbox if local_bbox is None else _bbox_union(local_bbox, char_local_bbox)
        previous_bbox = char_local_bbox
        widths.append(max(0.1, char_local_bbox[2] - char_local_bbox[0]))
    flush_fragment()
    return output


def _cluster_fragment_rows(
    fragments: list[_Fragment],
    median_height: float,
) -> list[_VisualRow]:
    """按中心线容差将不同文本片段聚成视觉行。"""

    tolerance = max(2.0, median_height * 0.5)
    grouped: list[list[_Fragment]] = []
    for fragment in sorted(
        fragments,
        key=lambda item: (_bbox_center_y(item.local_bbox), item.local_bbox[0]),
    ):
        center_y = _bbox_center_y(fragment.local_bbox)
        target_group: list[_Fragment] | None = None
        for group in grouped:
            group_center = sum(_bbox_center_y(item.local_bbox) for item in group) / len(group)
            if abs(center_y - group_center) <= tolerance:
                target_group = group
                break
        if target_group is None:
            grouped.append([fragment])
        else:
            target_group.append(fragment)

    rows: list[_VisualRow] = []
    for group in grouped:
        group.sort(key=lambda item: item.local_bbox[0])
        bbox = _bbox_union_many([item.local_bbox for item in group])
        rows.append(
            _VisualRow(
                fragments=group,
                center_y=sum(_bbox_center_y(item.local_bbox) for item in group) / len(group),
                bbox=bbox,
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
        # 强网格按竖线分隔后的实际文本占位计列，避免 OCR 左边界不稳定时漏掉有框表格。
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
    parse_mode: Literal["txt", "ocr"],
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
            if parse_mode == "txt":
                candidate_chars = [
                    char for line in source.lines if line.source_index in projection_line_indices for char in line.chars
                ]
                content = project_pdf_table_text(
                    candidate_chars,
                    candidate.bbox,
                    angle=candidate.angle,
                )
            else:
                content = _project_ocr_candidate(source, candidate)
                if candidate.angle != 0:
                    base_projection_score = _score_existing_ocr_projection(
                        source,
                        candidate,
                        content,
                    )
                    rotated_content, rotated_angle, rotated_score = _recognize_rotated_ocr_table(
                        source,
                        candidate,
                    )
                    if rotated_score > base_projection_score:
                        content = rotated_content
                        output_angle = rotated_angle
        except Exception as exc:
            # 单个表格的字符或局部 OCR 异常只撤销该候选，不能中止整页提取。
            logger.warning(
                f"Flash table projection failed and rolled back: parse_mode={parse_mode}, bbox={candidate.bbox}, error={exc}"
            )
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
    """合并显式候选成员与核心表格框内的其他方向文本行。"""

    line_indices = set(candidate.line_indices)
    if candidate.core_bbox is None:
        return line_indices
    for line in source.lines:
        if _point_in_bbox(
            (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
            candidate.core_bbox,
        ):
            line_indices.add(line.source_index)
    return line_indices


def _project_ocr_candidate(source: _PageSource, candidate: _TableCandidate) -> str:
    """将全页 OCR quad 平移并按表格角度旋转到局部投影坐标。"""

    scale = source.render_scale
    table_x0, table_y0, table_x1, table_y1 = candidate.bbox
    width_px = max(1, int(round((table_x1 - table_x0) * scale)))
    height_px = max(1, int(round((table_y1 - table_y0) * scale)))
    ocr_result: list[list[Any]] = []
    projection_line_indices = _candidate_projection_line_indices(source, candidate)
    for line in source.lines:
        if line.pixel_quad is None:
            continue
        if line.source_index not in projection_line_indices:
            continue
        local_points = [
            (
                point[0] - table_x0 * scale,
                point[1] - table_y0 * scale,
            )
            for point in line.pixel_quad
        ]
        rotated_points = _rotate_local_points(
            local_points,
            width_px,
            height_px,
            candidate.angle,
        )
        ocr_result.append(
            [
                [[float(x), float(y)] for x, y in rotated_points],
                (line.text, line.score),
            ]
        )
    table_size = (height_px, width_px) if candidate.angle in {90, 270} else (width_px, height_px)
    return project_ocr_table_text(ocr_result, table_size)


def _recognize_rotated_ocr_table(
    source: _PageSource,
    candidate: _TableCandidate,
) -> tuple[str, int, float]:
    """对旋转 OCR 表格分别尝试 90/270 度，选择投影可读性更高的结果。"""

    if source.bgr_image is None or source.ocr_model is None:
        return "", candidate.angle, 0.0
    ocr_model, run_ocr_inference = source.ocr_model
    scale = source.render_scale
    recognition_bbox = candidate.core_bbox or candidate.bbox
    x0, y0, x1, y1 = recognition_bbox
    image_height, image_width = source.bgr_image.shape[:2]
    left = max(0, min(image_width, int(math.floor(x0 * scale))))
    top = max(0, min(image_height, int(math.floor(y0 * scale))))
    right = max(0, min(image_width, int(math.ceil(x1 * scale))))
    bottom = max(0, min(image_height, int(math.ceil(y1 * scale))))
    if right <= left or bottom <= top:
        return "", candidate.angle, 0.0
    crop = source.bgr_image[top:bottom, left:right].copy()
    projection_line_indices = _candidate_projection_line_indices(source, candidate)
    # 远标题形成的矩形走廊中若存在未被候选接纳的正文，二次 OCR 前先遮白。
    for line in source.lines:
        if line.source_index in projection_line_indices or not _bbox_intersects(line.bbox, recognition_bbox):
            continue
        mask_left = max(0, min(crop.shape[1], int(math.floor(line.bbox[0] * scale)) - left))
        mask_top = max(0, min(crop.shape[0], int(math.floor(line.bbox[1] * scale)) - top))
        mask_right = max(0, min(crop.shape[1], int(math.ceil(line.bbox[2] * scale)) - left))
        mask_bottom = max(0, min(crop.shape[0], int(math.ceil(line.bbox[3] * scale)) - top))
        if mask_right > mask_left and mask_bottom > mask_top:
            crop[mask_top:mask_bottom, mask_left:mask_right] = 255
    best_content = ""
    best_angle = candidate.angle
    best_score = 0.0
    for quarter_turns, output_angle in ((1, 90), (3, 270)):
        rotated = np.ascontiguousarray(np.rot90(crop, k=quarter_turns))
        raw_result = run_ocr_inference(
            ocr_model.ocr,
            rotated,
            det=True,
            rec=True,
            tqdm_enable=False,
        )
        raw_ocr_result = raw_result[0] if raw_result else []
        filtered_ocr_result: list[list[Any]] = []
        for item in raw_ocr_result or []:
            if not item or len(item) < 2 or not item[1] or len(item[1]) < 2:
                continue
            text = str(item[1][0] or "").strip()
            try:
                confidence = float(item[1][1] or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if not text or confidence < _OCR_MIN_CONFIDENCE:
                continue
            filtered_ocr_result.append([item[0], (text, confidence)])
        if not filtered_ocr_result:
            continue
        content = project_ocr_table_text(
            filtered_ocr_result,
            (rotated.shape[1], rotated.shape[0]),
        )
        confidence_values = [
            float(item[1][1]) for item in filtered_ocr_result if item and len(item) >= 2 and item[1] and len(item[1]) >= 2
        ]
        mean_confidence = statistics.fmean(confidence_values) if confidence_values else 0.0
        score = mean_confidence * _valid_character_ratio(content) * _score_projected_table(content)
        if score > best_score:
            best_score = score
            best_content = content
            best_angle = output_angle
    return best_content, best_angle, best_score


def _score_existing_ocr_projection(
    source: _PageSource,
    candidate: _TableCandidate,
    content: str,
) -> float:
    """使用候选原 OCR 均值置信度、字符有效率和空间结构评分现有投影。"""

    projection_line_indices = _candidate_projection_line_indices(source, candidate)
    confidence_values = [
        line.score
        for line in source.lines
        if line.source_index in projection_line_indices and line.pixel_quad is not None and line.score >= _OCR_MIN_CONFIDENCE
    ]
    mean_confidence = statistics.fmean(confidence_values) if confidence_values else 0.0
    return mean_confidence * _valid_character_ratio(content) * _score_projected_table(content)


def _score_projected_table(content: str) -> float:
    """根据非空行数和具有明显列间隔的行数评分投影结果。"""

    if not content:
        return 0.0
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return 0.0
    column_lines = sum(bool(re.search(r"\S\s{2,}\S", line)) for line in lines)
    return min(1.0, len(lines) / 3.0) * (1.0 + column_lines / len(lines))


def _valid_character_ratio(content: str) -> float:
    """计算 OCR 内字母、数字或 CJK 字符占所有非空字符的比例。"""

    characters = [char for char in content if not char.isspace()]
    if not characters:
        return 0.0
    valid = sum(char.isalnum() or "\u3400" <= char <= "\u9fff" for char in characters)
    return valid / len(characters)


def _build_text_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """将剩余文本行按位置、行距和行高聚成统一 text block。"""

    blocks: list[dict[str, Any]] = []
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        # OCR/PDF 原始流顺序不可靠，先按正向局部视觉坐标排序再构造有向行连接。
        line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        angle_lines = [item[0] for item in line_geometry]
        local_bboxes = [item[1] for item in line_geometry]
        heights = [bbox[3] - bbox[1] for bbox in local_bboxes if bbox[3] > bbox[1]]
        median_height = statistics.median(heights) if heights else 1.0
        parents = list(range(len(angle_lines)))

        def find(index: int) -> int:
            """查找文本行所属连通分量的根节点。"""

            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            """合并符合段落几何条件的相邻文本行。"""

            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left_index, left_bbox in enumerate(local_bboxes):
            left_height = left_bbox[3] - left_bbox[1]
            for right_index in range(left_index + 1, len(angle_lines)):
                right_bbox = local_bboxes[right_index]
                if right_bbox[1] < left_bbox[1]:
                    continue
                right_height = right_bbox[3] - right_bbox[1]
                if min(left_height, right_height) <= 0:
                    continue
                height_ratio = max(left_height, right_height) / min(left_height, right_height)
                if height_ratio > 2.0:
                    continue
                vertical_gap = right_bbox[1] - left_bbox[3]
                if vertical_gap < -0.25 * median_height or vertical_gap > 1.8 * median_height:
                    continue
                overlap_ratio = _bbox_axis_overlap_ratio(left_bbox, right_bbox, axis="x")
                left_edge_delta = abs(left_bbox[0] - right_bbox[0])
                if overlap_ratio < 0.5 and left_edge_delta > 1.5 * median_height:
                    continue
                if _connection_crosses_table(
                    angle_lines[left_index].bbox,
                    angle_lines[right_index].bbox,
                    table_bboxes,
                ):
                    continue
                union(left_index, right_index)

        components: dict[int, list[int]] = {}
        for index in range(len(angle_lines)):
            components.setdefault(find(index), []).append(index)
        for indices in components.values():
            indices.sort(key=lambda index: (local_bboxes[index][1], local_bboxes[index][0]))
            component_lines = [angle_lines[index] for index in indices]
            content = _merge_text_line_content([line.text for line in component_lines])
            if not content:
                continue
            blocks.append(
                {
                    "type": "text",
                    "bbox": _bbox_union_many([line.bbox for line in component_lines]),
                    "angle": angle,
                    "content": content,
                }
            )
    return blocks


def _merge_text_line_content(line_texts: Sequence[str]) -> str:
    """按 Hybrid 语言与行末连字规则折叠普通文本行。"""

    normalized_lines = [str(text or "").strip() for text in line_texts]
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
    if not isinstance(content, str) or not content.strip():
        return None
    return {
        "type": "table" if block.get("type") == "table" else "text",
        "bbox": [
            round(bbox[0] / page_width, 6),
            round(bbox[1] / page_height, 6),
            round(bbox[2] / page_width, 6),
            round(bbox[3] / page_height, 6),
        ],
        "angle": int(block.get("angle", 0) or 0) % 360,
        "content": content,
    }


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


def _rotate_local_points(
    points: list[tuple[float, float]],
    width: float,
    height: float,
    angle: int,
) -> list[tuple[float, float]]:
    """将表格局部 OCR 点按标准方向旋转到正向坐标。"""

    if angle == 270:
        return [(height - y, x) for x, y in points]
    if angle == 90:
        return [(y, width - x) for x, y in points]
    if angle == 180:
        return [(width - x, height - y) for x, y in points]
    return points


def _merge_axis_lines(lines: list[_AxisLine], tolerance: float) -> list[_AxisLine]:
    """合并方向相同、中心线相近且范围相接的图像线段。"""

    merged: list[_AxisLine] = []
    for line in sorted(lines, key=lambda item: (item.orientation, item.bbox[1], item.bbox[0])):
        target: _AxisLine | None = None
        for existing in merged:
            if existing.orientation != line.orientation:
                continue
            if line.orientation == "horizontal":
                same_axis = abs(_bbox_center_y(existing.bbox) - _bbox_center_y(line.bbox)) <= tolerance
                touching = line.bbox[0] <= existing.bbox[2] + tolerance and line.bbox[2] >= existing.bbox[0] - tolerance
            else:
                same_axis = abs(_bbox_center_x(existing.bbox) - _bbox_center_x(line.bbox)) <= tolerance
                touching = line.bbox[1] <= existing.bbox[3] + tolerance and line.bbox[3] >= existing.bbox[1] - tolerance
            if same_axis and touching:
                target = existing
                break
        if target is None:
            merged.append(line)
        else:
            target.bbox = _bbox_union(target.bbox, line.bbox)
            target.width = max(target.width, line.width)
    return merged


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


def _median_line_height(lines: list[_LineItem]) -> float:
    """返回有效文本行高的中位数。"""

    heights = [line.bbox[3] - line.bbox[1] for line in lines if line.bbox[3] > line.bbox[1]]
    return float(statistics.median(heights)) if heights else 1.0


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
