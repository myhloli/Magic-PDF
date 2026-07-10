# Copyright (c) Opendatalab. All rights reserved.
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image
from loguru import logger
from mineru.backend.local_model_runtime import AtomModelSingleton, HybridLocalModelContextSingleton
from mineru.backend.utils.runtime_utils import exclude_progress_bar_idle_time
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.model_utils import clean_memory
from mineru.utils.pdf_image_tools import load_images_from_pdf_bytes_range
from tqdm import tqdm

from ...utils.image_payload import ImagePayloadCache
from ...utils.pdf_document import PDFDocument
from ...types import PageInfo, BlockType, BBox, NOT_EXTRACT_TYPES
from ...utils.config_reader import get_processing_window_size


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 让mps可以fallback

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8

VLM_LAYOUT_LABEL_MAP = {
    "abstract": BlockType.TEXT,
    "algorithm": BlockType.CODE,
    "aside_text": BlockType.ASIDE_TEXT,
    "chart": BlockType.CHART,
    "content": BlockType.INDEX,
    "display_formula": BlockType.EQUATION,
    "doc_title": BlockType.DOC_TITLE,
    "figure_title": BlockType.CAPTION,
    "footer": BlockType.FOOTER,
    "footer_image": BlockType.FOOTER,
    "footnote": BlockType.PAGE_FOOTNOTE,
    "formula_number": BlockType.FORMULA_NUMBER,
    "header": BlockType.HEADER,
    "header_image": BlockType.HEADER,
    "image": BlockType.IMAGE,
    "number": BlockType.PAGE_NUMBER,
    "paragraph_title": BlockType.PARAGRAPH_TITLE,
    "reference_content": BlockType.REF_TEXT,
    "seal": BlockType.IMAGE,
    "table": BlockType.TABLE,
    "text": BlockType.TEXT,
    "vertical_text": BlockType.TEXT,
    "vision_footnote": BlockType.FOOTNOTE,
}

PIPELINE_DET_TYPE = {
    BlockType.TEXT,
    BlockType.CODE,
    BlockType.ASIDE_TEXT,
    BlockType.INDEX,
    BlockType.DOC_TITLE,
    BlockType.CAPTION,
    BlockType.FOOTER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.FORMULA_NUMBER,
    BlockType.HEADER,
    BlockType.PAGE_NUMBER,
    BlockType.PARAGRAPH_TITLE,
    BlockType.REF_TEXT,
    BlockType.FOOTNOTE,
}
VLM_TXT_DET_TYPE = NOT_EXTRACT_TYPES
VLM_OCR_DET_TYPE = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}


def _load_vlm_runtime() -> dict[str, Any]:
    """按需加载 VLM runtime 组件，确保只有 high/extra_high 路径触发 VLM 依赖。"""
    from ...model.vlm.runtime import (
        ModelSingleton,
        _get_model_async,
        _maybe_enable_serial_execution,
        aio_predictor_execution_guard,
        predictor_execution_guard,
    )

    return {
        "ModelSingleton": ModelSingleton,
        "_get_model_async": _get_model_async,
        "_maybe_enable_serial_execution": _maybe_enable_serial_execution,
        "aio_predictor_execution_guard": aio_predictor_execution_guard,
        "predictor_execution_guard": predictor_execution_guard,
    }


@dataclass(frozen=True)
class _ProcessingWindow:
    """记录单个 Hybrid 处理窗口的页码范围，统一同步和异步入口的窗口计算。"""
    index: int
    total: int
    start: int
    end: int


def _build_processing_windows(page_count: int, configured_window_size: int) -> list[_ProcessingWindow]:
    """根据页数和配置窗口大小生成稳定的 Hybrid 分段处理计划。"""
    effective_window_size = min(page_count, configured_window_size) if page_count else 0
    if effective_window_size <= 0:
        return []

    total_windows = (page_count + effective_window_size - 1) // effective_window_size
    return [
        _ProcessingWindow(
            index=window_index,
            total=total_windows,
            start=window_start,
            end=min(page_count - 1, window_start + effective_window_size - 1),
        )
        for window_index, window_start in enumerate(range(0, page_count, effective_window_size))
    ]


def _log_processing_window_plan(page_count: int, configured_window_size: int, total_windows: int) -> None:
    """输出 Hybrid 分段处理计划日志，避免同步和异步入口文案漂移。"""
    logger.info(
        f"Hybrid processing-window run. page_count={page_count}, "
        f"window_size={configured_window_size}, total_windows={total_windows}"
    )


def _log_processing_window(window: _ProcessingWindow, page_count: int, image_count: int) -> None:
    """输出单个 Hybrid 处理窗口的页码范围日志。"""
    logger.info(
        f"Hybrid processing window {window.index + 1}/{window.total}: "
        f"pages {window.start + 1}-{window.end + 1}/{page_count} "
        f"({image_count} pages)"
    )


def _close_images(images_list: list[dict[str, Any]]) -> None:
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def _normalize_page_size(page_image: Any) -> tuple[int, int]:
    """从PIL或numpy图像中读取页面宽高，供归一化bbox还原为像素bbox。"""
    if hasattr(page_image, "size"):
        return page_image.size

    height, width = page_image.shape[:2]
    return width, height


def _bbox_to_pixel_bbox(bbox: BBox | None, page_size: tuple[int, int]) -> BBox | None:
    """将归一化或像素bbox统一成像素bbox，异常bbox返回None。"""
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    width, height = page_size
    if all(0.0 <= value <= 1.0 for value in [x0, y0, x1, y1]):
        x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height

    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _normalize_layout_bbox_to_unit(bbox: BBox | None, page_size: tuple[int, int]) -> list[float] | None:
    """将 layout 像素 bbox 归一化为 VLM ContentBlock 需要的 0-1 坐标。"""
    pixel_bbox = _bbox_to_pixel_bbox(bbox, page_size)
    if pixel_bbox is None:
        return None

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return None

    x0, y0, x1, y1 = pixel_bbox
    unit_bbox = [
        round(max(0.0, min(1.0, float(x0) / page_width)), 3),
        round(max(0.0, min(1.0, float(y0) / page_height)), 3),
        round(max(0.0, min(1.0, float(x1) / page_width)), 3),
        round(max(0.0, min(1.0, float(y1) / page_height)), 3),
    ]
    if unit_bbox[2] <= unit_bbox[0] or unit_bbox[3] <= unit_bbox[1]:
        return None
    return unit_bbox


def _layout_item_to_content_block(layout_item: dict[str, Any], page_size: tuple[int, int]) -> dict | None:
    """将本地 layout 小模型检测项转换为 mineru-vl-utils 的 ContentBlock。"""
    label = layout_item.get("label") or layout_item.get("type")

    block_type = VLM_LAYOUT_LABEL_MAP.get(str(label))
    if block_type is None:
        return None

    bbox = _normalize_layout_bbox_to_unit(layout_item.get("bbox"), page_size)
    if bbox is None:
        return None

    content_block = {
        "type": block_type,
        "bbox": bbox,
        "angle": layout_item.get("angle", 0),
    }

    if block_type == BlockType.IMAGE and label == "seal":
        content_block["sub_type"] = "seal"

    return content_block


def _build_vl_style_layout_blocks(
    images_layout_res: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
    effort: Literal["medium", "high", "xhigh"] = "high",
) -> list[list[Any]]:
    """按页构造 Hybrid high 模式传给 VLM 的外部 layout blocks。"""
    blocks_list: list[list[Any]] = []
    for layout_res, image in zip(images_layout_res, images_pil_list):
        page_size = _normalize_page_size(image)
        page_blocks = []
        for layout_item in layout_res:
            content_block = _layout_item_to_content_block(layout_item, page_size)
            if content_block is not None:
                page_blocks.append(content_block)
        blocks_list.append(page_blocks)
    return blocks_list


def doc_analyze(
    pdf_bytes: bytes,
    effort: Literal["medium", "high", "xhigh"] = "high",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
) -> tuple[list[PageInfo], list[list[dict[str, Any]]]]:
    batch_ratio = 2
    pdf_doc = PDFDocument(pdf_bytes)
    parse_mode = pdf_doc.classify()

    middle_json: list[PageInfo] = []
    model_list: list[list[dict[str, Any]]] = []
    doc_closed = False

    try:
        page_count = pdf_doc.page_count
        configured_window_size = get_processing_window_size(default=64)
        windows = _build_processing_windows(page_count, configured_window_size)
        _log_processing_window_plan(page_count, configured_window_size, len(windows))

        hybrid_model_singleton = HybridLocalModelContextSingleton()
        hybrid_model = hybrid_model_singleton.get_model()
        atom_model_manager = AtomModelSingleton()

        if effort in ["high", "xhigh"]:
            vlm_runtime = _load_vlm_runtime()
            vlm_backend = get_vlm_engine(inference_engine="auto", is_async=False)
            vlm_predictor = vlm_runtime["ModelSingleton"]().get_model(backend=vlm_backend)
            vlm_predictor = vlm_runtime["_maybe_enable_serial_execution"](vlm_predictor, vlm_backend)
        else:
            vlm_predictor = None

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None

        try:
            for window in windows:
                images_list = load_images_from_pdf_bytes_range(
                    pdf_bytes=pdf_bytes,
                    start_page_id=window.start,
                    end_page_id=window.end,
                    image_type="pil_img",
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    _log_processing_window(window, page_count, len(images_pil_list))

                    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
                    images_layout_res = hybrid_model.layout_model.batch_predict(
                        images_pil_list,
                        batch_size=min(8, batch_ratio * LAYOUT_BASE_BATCH_SIZE)
                    )

                    vl_style_layout_blocks = _build_vl_style_layout_blocks(images_layout_res, images_pil_list, effort)

                    if effort == "medium":
                        pass
                    elif effort == "high":
                        pass
                    elif effort == "xhigh":
                        pass
                    else:
                        raise ValueError(f"Unsupported analyze effort: {effort}")

                    model_list.extend(window_model_list)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_pages(
                        middle_json,
                        window_model_list,
                        images_list,
                        pdf_doc,
                        page_cvt_fn=blocks_to_page_info,
                        page_start_index=window.start,
                        page_index_map=page_index_map,
                        _ocr_enable=_ocr_enable,
                        use_vlm_text_content=use_vlm_text_content,
                        progress_bar=progress_bar,
                        image_cache=image_cache,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"processing-window infer finished, cost: {infer_time}, speed: {round(len(model_list) / infer_time, 3)} page/s"
            )

        pdf_doc.close()
        doc_closed = True
        clean_memory(hybrid_model.device)
        return middle_json, model_list
    finally:
        if not doc_closed:
            pdf_doc.close()


if __name__ == '__main__':
    pdf_path = "/Users/myhloli/pdf/截断合并/demo1-2.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    middle_json, model_list = doc_analyze(pdf_bytes, effort="medium", image_analysis=True)
    logger.info(f"middle_json: {middle_json}")
    logger.info(f"model_list: {model_list}")
