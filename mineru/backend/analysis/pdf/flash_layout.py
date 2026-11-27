# Copyright (c) Opendatalab. All rights reserved.
"""为 Flash/TXT 的告警页批量运行布局模型，按请求内物理页号提交完整结果。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
import html
import re
from typing import TYPE_CHECKING, Any

from docvortex.document.pdf.visuals import attach_visual_block_images
from docvortex.geometry import normalize_bbox
from loguru import logger

from ....model.runtime.execution import acquire_document, local_model_stage, release_document
from ....model.runtime.memory import clean_memory
from .images import load_images_from_pdf_bytes_range

if TYPE_CHECKING:
    from docvortex.document.pdf import PDFDocument
    from PIL import Image
    from ....model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX
    from ....model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel

    LayoutModel = PPDocLayoutV2LayoutModel | PPDocLayoutV2LayoutModelONNX


def _load_layout_model() -> tuple[str, LayoutModel]:
    """首次命中时复用当前设备的布局原子模型，避免初始化整个 Hybrid 识别栈。"""
    from ....model.runtime.contracts import AtomicModelName
    from ....model.runtime.device_state import resolve_model_device
    from ....model.runtime.hybrid import AtomModelSingleton

    device = resolve_model_device().model_device
    return device, AtomModelSingleton().get_atom_model(AtomicModelName.Layout)


def _predict_batch(layout_model: LayoutModel, device: str, images: list[Image.Image]) -> list[list[dict]]:
    """在共享设备锁内推理，并检查预测与输入物理页一一对应。"""
    with local_model_stage(device):
        predictions = layout_model.batch_predict(images, batch_size=len(images))
    if len(predictions) != len(images):
        raise ValueError("Flash layout prediction page count mismatch")
    return predictions


def _predict_pages(
    layout_model: LayoutModel, device: str, rendered: list[tuple[int, dict[str, Any]]]
) -> Iterator[tuple[tuple[int, dict[str, Any]], list[dict]]]:
    """批任务异常时单独重试各页，保住同批其他页的可用结果。"""
    images = [image["img_pil"] for _, image in rendered]
    try:
        predictions = _predict_batch(layout_model, device, images)
    except Exception as exc:
        if len(rendered) == 1:
            logger.warning("Flash TXT kept native layout for page {}: {}", rendered[0][0] + 1, exc)
            return
        logger.warning("Flash layout batch failed; retrying {} pages separately: {}", len(rendered), exc)
        for entry in rendered:
            yield from _predict_pages(layout_model, device, [entry])
    else:
        yield from zip(rendered, predictions, strict=True)


_CONTENT_BLOCK_TYPES = frozenset({"text", "paragraph_title", "doc_title", "code", "list", "table"})
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMPTY_PP_MIN_NATIVE_CHARS = 50
_UNCOVERED_MIN_CHARS = 150
_UNCOVERED_MIN_RATIO = 0.30
_BLOCK_COVERED_THRESHOLD = 0.50
_REVERSAL_MIN_NATIVE_CONF = 0.60
_REVERSAL_MAX_PP_CONF = 0.40
_REVERSAL_MIN_DROP = 0.25


def _visible_nonws_chars(block: dict[str, Any]) -> int:
    """统计块内可见非空白字符数；表格去除 HTML 标签，递归展开嵌套列表与行内 span。"""
    content = block.get("content")
    if isinstance(content, list):
        return sum(_visible_nonws_chars(item) for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        return 0
    text = html.unescape(_HTML_TAG_RE.sub(" ", content)) if block.get("type") == "table" and "<" in content else content
    return sum(not ch.isspace() for ch in text)


def _union_coverage_unit(
    target: tuple[float, float, float, float], covers: list[tuple[float, float, float, float]]
) -> float:
    """按 x 切片计算多个 PP 区域并集对目标块面积的精确覆盖，不重复计重叠部分。"""
    intersections = [
        (max(target[0], box[0]), max(target[1], box[1]), min(target[2], box[2]), min(target[3], box[3]))
        for box in covers
        if min(target[2], box[2]) > max(target[0], box[0]) and min(target[3], box[3]) > max(target[1], box[1])
    ]
    if not intersections:
        return 0.0
    x_values = sorted({value for box in intersections for value in (box[0], box[2])})
    area = 0.0
    for left, right in zip(x_values, x_values[1:]):
        spans = sorted((box[1], box[3]) for box in intersections if box[0] <= left < box[2])
        end = target[1]
        for start, stop in spans:
            area += (right - left) * max(0.0, stop - max(end, start))
            end = max(end, stop)
    return area


def _check_pp_replacement_safety(
    native_blocks: list[dict[str, Any]],
    pp_blocks: list[dict[str, Any]],
    *,
    native_confidence: float | None = None,
) -> str | None:
    """拦截 PP 严重漏检正文或对边界页造成反向重度碎裂的替换，返回拒绝原因或 None。"""
    total_native_chars = 0
    native_content: list[tuple[tuple[float, float, float, float] | None, int]] = []
    for block in native_blocks:
        if block.get("type") not in _CONTENT_BLOCK_TYPES:
            continue
        chars = _visible_nonws_chars(block)
        if chars <= 0:
            continue
        total_native_chars += chars
        native_content.append((normalize_bbox(block.get("bbox")), chars))

    if not pp_blocks and total_native_chars >= _EMPTY_PP_MIN_NATIVE_CHARS:
        return f"empty_pp_output(native_chars={total_native_chars})"

    if total_native_chars >= _UNCOVERED_MIN_CHARS:
        pp_boxes = [box for block in pp_blocks if (box := normalize_bbox(block.get("bbox"))) is not None]
        uncovered_chars = 0
        for box, chars in native_content:
            if box is None:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area <= 0:
                continue
            if _union_coverage_unit(box, pp_boxes) / area < _BLOCK_COVERED_THRESHOLD:
                uncovered_chars += chars
        uncovered_ratio = uncovered_chars / total_native_chars
        if uncovered_chars >= _UNCOVERED_MIN_CHARS and uncovered_ratio >= _UNCOVERED_MIN_RATIO:
            return f"uncovered_text_loss({uncovered_chars}/{total_native_chars}={uncovered_ratio:.0%})"

    if native_confidence is not None and native_confidence > _REVERSAL_MIN_NATIVE_CONF and pp_blocks:
        try:
            from docvortex.analyzers.native.pdf.layout_risk_scoring import assess_layout_pages
        except ImportError:
            return None
        pp_assessment = assess_layout_pages([pp_blocks])[0]
        if (
            pp_assessment.confidence <= _REVERSAL_MAX_PP_CONF
            and pp_assessment.confidence <= native_confidence - _REVERSAL_MIN_DROP
            and pp_assessment.domain_points.get("fragmentation", 0) >= 60
        ):
            return f"severe_fragmentation_reversal(native={native_confidence:.2f}->pp={pp_assessment.confidence:.2f})"
    return None


def replace_flash_layout_pages(
    document: PDFDocument,
    native_pages: list[list[dict]],
    page_indices: list[int],
    batch_size: int,
    *,
    native_confidences: dict[int, float] | None = None,
) -> set[int]:
    """只渲染命中的物理页；回填、防退化校验及素材裁切全部通过后才替换该页的原生块。"""
    from .flash_layout_fill import fill_flash_layout_page

    try:
        device, layout_model = _load_layout_model()
    except Exception as exc:
        logger.warning("Flash TXT kept native layout for {} pages; layout model unavailable: {}", len(page_indices), exc)
        return set()
    completed = set()
    acquire_document(device)
    try:
        for start in range(0, len(page_indices), batch_size):
            with ExitStack() as resources:
                rendered = []
                for index in page_indices[start : start + batch_size]:
                    try:
                        (image,) = load_images_from_pdf_bytes_range(
                            document.bytes, document=document, start_page_id=index, end_page_id=index
                        )
                        resources.callback(image["img_pil"].close)
                        rendered.append((index, image))
                    except Exception as exc:
                        logger.warning("Flash TXT kept native layout for page {} after render failure: {}", index + 1, exc)
                if not rendered:
                    continue
                for (index, image), items in _predict_pages(layout_model, device, rendered):
                    try:
                        blocks = fill_flash_layout_page(document[index], items, image["img_pil"].size)
                    except Exception as exc:
                        logger.warning("Flash TXT kept native layout for page {} after refill failure: {}", index + 1, exc)
                        continue
                    native_conf = native_confidences.get(index) if native_confidences else None
                    if reject_reason := _check_pp_replacement_safety(
                        native_pages[index], blocks, native_confidence=native_conf
                    ):
                        logger.info(
                            "Flash TXT kept native layout for page {} after PP quality guard: {}",
                            index + 1,
                            reject_reason,
                        )
                        continue
                    try:
                        attach_visual_block_images([blocks], [image], page_start_index=index)
                    except Exception as exc:
                        logger.warning("Flash TXT kept native layout for page {} after crop failure: {}", index + 1, exc)
                        continue
                    native_pages[index] = blocks
                    completed.add(index)
    finally:
        release_document(device, clean_memory)
    return completed
