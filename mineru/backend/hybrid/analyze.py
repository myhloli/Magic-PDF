# Copyright (c) Opendatalab. All rights reserved.
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from mineru.backend.local_model_runtime import HybridLocalModelContextSingleton, HybridLocalModelContext
from mineru.backend.utils.boxbase import calculate_overlap_area_2_minbox_area_ratio
from mineru.backend.utils.formula_number import optimize_hybrid_formula_number_blocks
from mineru.backend.utils.runtime_utils import exclude_progress_bar_idle_time
from mineru.cli_old.common import read_fn
from mineru.utils.bbox_utils import normalize_to_int_bbox
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.model_utils import clean_memory, crop_img
from mineru.utils.ocr_utils import (
    get_adjusted_mfdetrec_res,
    mask_formula_regions_for_ocr_det,
    sorted_boxes,
    merge_det_boxes,
    update_det_boxes,
    get_ocr_result_list,
    OcrConfidence,
    get_rotate_crop_image_for_text_rec,
)
from mineru.utils.pdf_image_tools import load_images_from_pdf_bytes_range, get_crop_np_img
from tqdm import tqdm

from ...utils.image_payload import ImagePayloadCache
from ...utils.pdf_document import PDFDocument, PDFPage
from ...types import PageInfo, BlockType, BBox, NOT_EXTRACT_TYPES
from ...utils.config_reader import get_processing_window_size


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 让mps可以fallback

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8
BATCH_RATIO = 2

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


@dataclass
class _OcrDetCrop:
    """保存一次 OCR det 裁剪的中间数据，避免用裸 tuple 传递阶段状态。"""

    bgr_image: Any
    useful_list: list[Any]
    adjusted_mfdetrec_res: list[Any]
    page_ocr_res_list: list[dict[str, Any]]



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


def _get_window_pdf_pages(pdf_doc: PDFDocument, window: _ProcessingWindow) -> list[PDFPage]:
    """按窗口闭区间获取对应的 PDFPage 对象，供窗口内后续处理复用。"""
    return [pdf_doc[page_idx] for page_idx in range(window.start, window.end + 1)]


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


def _get_crop_table_img(
        np_img: np.ndarray,
        table_res_bbox: BBox,
        scale: float= 1,
) -> np.ndarray:
    """按指定缩放裁剪表格图，保持 medium 表格处理只使用当前文件窗口图像。"""
    bbox = normalize_to_int_bbox([float(v) / float(scale) for v in table_res_bbox])
    if bbox is None:
        return np_img[0:0, 0:0]
    return get_crop_np_img(bbox, np_img, scale=scale)


def _collect_table_items(
    images_layout_res: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> list[dict[str, Any]]:
    table_items = []
    for layout_res, np_img in zip(images_layout_res, np_images):
        for table_res in layout_res:
            if table_res.get("label") != "table":
                continue
            table_img = _get_crop_table_img(np_img=np_img, table_res_bbox=table_res["bbox"])
            if table_img.size == 0:
                continue
            table_items.append(
                {
                    "table_img": table_img,
                    "layout_item": table_res,
                }
            )
    return table_items


def _apply_table_rotate_labels(
    table_items: list[dict[str, Any]],
    rotate_labels: list[str],
) -> None:
    """按分类输入顺序将表格旋转角写回原始 layout 检测项。"""
    if len(rotate_labels) != len(table_items):
        raise ValueError("Hybrid table orientation result count mismatch")

    for table_item, rotate_label in zip(table_items, rotate_labels):
        table_item["layout_item"]["angle"] = int(rotate_label or "0")


def _build_vl_style_layout_blocks(
    images_layout_res: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
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


def _build_formula_inputs(images_layout_res: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """构造完整 MFD/MFR 输入，保留全部行内和行间公式框。"""
    formula_inputs = []
    for layout_res in images_layout_res:
        page_formula_inputs = []
        for res in layout_res:
            label = res.get("label")
            if label not in ["inline_formula", "display_formula"]:
                continue
            bbox = res.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            page_formula_inputs.append(
                {
                    "label": label,
                    "bbox": list(bbox),
                    "score": float(res.get("score", 0.0)),
                    # layout 只提供公式位置；未运行 MFR 的 high/xhigh OCR 必须保留空 LaTeX。
                    "latex": "",
                }
            )
        formula_inputs.append(page_formula_inputs)
    return formula_inputs


def _split_formula_results(
    images_formula_list: list[list[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """按原始标签拆分 MFR 结果，避免行间公式进入 inline sidecar。"""
    inline_formula_list = []
    display_formula_list = []
    for page_formula_list in images_formula_list:
        inline_formula_list.append(
            [formula for formula in page_formula_list if formula.get("label") == "inline_formula"]
        )
        display_formula_list.append(
            [formula for formula in page_formula_list if formula.get("label") == "display_formula"]
        )
    return inline_formula_list, display_formula_list


def _apply_medium_display_formula_results(
    model_list: list[list[dict[str, Any]]],
    display_formula_list: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
) -> None:
    """将 medium 行间公式 LaTeX 按页和 bbox 回填到对应 equation 块。"""
    for page_idx, (page_model_list, page_display_formula_list, page_image) in enumerate(
        zip(model_list, display_formula_list, images_pil_list)
    ):
        page_size = _normalize_page_size(page_image)
        equation_blocks_by_bbox: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for block in page_model_list:
            if block.get("type") != BlockType.EQUATION:
                continue
            block_bbox = block.get("bbox")
            if block_bbox is None or len(block_bbox) != 4:
                continue
            equation_blocks_by_bbox.setdefault(tuple(float(value) for value in block_bbox), []).append(block)

        for formula in page_display_formula_list:
            normalized_bbox = _normalize_layout_bbox_to_unit(formula.get("bbox"), page_size)
            if normalized_bbox is None:
                continue
            matched_blocks = equation_blocks_by_bbox.get(tuple(normalized_bbox), [])
            if len(matched_blocks) != 1:
                raise ValueError(
                    "Hybrid medium display formula must match exactly one equation block: "
                    f"page_idx={page_idx}, bbox={normalized_bbox}, matches={len(matched_blocks)}"
                )
            matched_blocks[0]["content"] = formula.get("latex", "")


def _build_ocr_det_type_and_mfr_enable(
    parse_mode: Literal["txt", "ocr"],
    effort: Literal["medium", "high", "xhigh"],
) -> tuple[set[str], bool]:
    """返回 OCR 检测块类型，以及是否需要执行小模型公式识别。"""
    if parse_mode not in ("txt", "ocr"):
        raise ValueError(f"Unsupported parse mode: {parse_mode}")
    if effort not in ("medium", "high", "xhigh"):
        raise ValueError(f"Unsupported analyze effort: {effort}")

    if effort == "medium":
        return PIPELINE_DET_TYPE, True
    if parse_mode == "txt":
        return VLM_TXT_DET_TYPE, True
    return VLM_OCR_DET_TYPE, False


def _formula_item_to_pixel_bbox(item: dict[str, Any]) -> list[int] | None:
    bbox = item.get("bbox")
    if bbox is not None and len(bbox) == 4:
        return [int(float(v)) for v in bbox]
    return None


def _set_temp_pixel_bbox(res: dict[str, Any], pixel_bbox: list[int]) -> None:
    """临时切换为像素 bbox，便于复用已有裁剪逻辑。"""
    res["_normalized_bbox"] = list(res["bbox"])
    res["bbox"] = pixel_bbox


def _restore_normalized_bbox(res: dict[str, Any]) -> None:
    """恢复归一化 bbox，避免 OCR det 过程污染 Hybrid 输出。"""
    normalized_bbox = res.pop("_normalized_bbox", None)
    if normalized_bbox is not None:
        res["bbox"] = normalized_bbox


def _collect_ocr_det_crops(
    np_images: list[Any],
    model_list: list[list[dict[str, Any]]],
    mfd_res: list[Any],
    ocr_det_type: set[str],
) -> tuple[list[list[dict[str, Any]]], list[_OcrDetCrop]]:
    """收集 OCR det 需要处理的裁剪图，并为每页预建 sidecar 结果列表。"""
    ocr_res_list: list[list[dict[str, Any]]] = []
    crops: list[_OcrDetCrop] = []

    for np_image, page_mfd_res, page_results in zip(np_images, mfd_res, model_list):
        page_ocr_res_list: list[dict[str, Any]] = []
        ocr_res_list.append(page_ocr_res_list)
        img_height, img_width = np_image.shape[:2]
        for res in page_results:
            if res["type"] not in ocr_det_type:
                continue
            x0 = max(0, int(res["bbox"][0] * img_width))
            y0 = max(0, int(res["bbox"][1] * img_height))
            x1 = min(img_width, int(res["bbox"][2] * img_width))
            y1 = min(img_height, int(res["bbox"][3] * img_height))
            if x1 <= x0 or y1 <= y0:
                continue
            _set_temp_pixel_bbox(res, [x0, y0, x1, y1])
            try:
                new_image, useful_list = crop_img(res, np_image, crop_paste_x=50, crop_paste_y=50)
            finally:
                _restore_normalized_bbox(res)
            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(page_mfd_res, useful_list)
            bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)  # type: ignore
            bgr_image = mask_formula_regions_for_ocr_det(bgr_image, adjusted_mfdetrec_res)
            crops.append(
                _OcrDetCrop(
                    bgr_image=bgr_image,
                    useful_list=useful_list,
                    adjusted_mfdetrec_res=adjusted_mfdetrec_res,
                    page_ocr_res_list=page_ocr_res_list,
                )
            )

    return ocr_res_list, crops


def _normalize_batch_ocr_det_boxes(dt_boxes: Any, adjusted_mfdetrec_res: list[Any]) -> list[Any]:
    """对 batch OCR det 的检测框排序、合并，并按公式位置修正。"""
    if dt_boxes is None or len(dt_boxes) == 0:
        return []

    dt_boxes_sorted = sorted_boxes(dt_boxes)
    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []
    if dt_boxes_merged and adjusted_mfdetrec_res:
        return update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
    return dt_boxes_merged


def _append_ocr_det_result(
    crop: _OcrDetCrop,
    ocr_res: Any,
    need_rec_img: bool,
) -> None:
    """将 OCR det 原始框转换为 Hybrid ocr_text sidecar 并写回对应页。"""
    if not ocr_res:
        return
    ocr_result_list = get_ocr_result_list(
        ocr_res,
        crop.useful_list,
        need_rec_img,
        crop.bgr_image,
    )
    crop.page_ocr_res_list.extend(ocr_result_list)


def _ocr_det(
    local_model_context: HybridLocalModelContext,
    np_images: list[np.ndarray],
    model_list: list[list[dict[str, Any]]],
    mfd_res: list[Any],
    need_rec_img: bool,
    ocr_det_type: set[str],
) -> list[list[dict[str, Any]]]:
    """执行 Hybrid OCR det sidecar 生成，按运行时配置选择单图或 batch 模式。"""
    ocr_res_list, crops = _collect_ocr_det_crops(np_images, model_list, mfd_res, ocr_det_type)

    if crops:
        batch_images = [crop.bgr_image for crop in crops]
        det_batch_size = min(len(batch_images), BATCH_RATIO * OCR_DET_BASE_BATCH_SIZE)
        batch_results = local_model_context.ocr_model.text_detector.batch_predict(
            batch_images,
            det_batch_size,
            tqdm_enable=True,
            tqdm_desc="OCR-det",
        )

        for crop, (dt_boxes, _) in zip(crops, batch_results):
            dt_boxes_final = _normalize_batch_ocr_det_boxes(dt_boxes, crop.adjusted_mfdetrec_res)
            if dt_boxes_final:
                ocr_res = [box.tolist() if hasattr(box, "tolist") else box for box in dt_boxes_final]
                _append_ocr_det_result(crop, ocr_res, need_rec_img)
    return ocr_res_list


def _collect_ocr_rec_inputs(
    ocr_res_list: list[list[dict[str, Any]]],
) -> tuple[list[tuple[list[dict[str, Any]], dict[str, Any]]], list[Any]]:
    """收集需要 OCR rec 的裁剪图，同时从 sidecar 中移除临时图像对象。"""
    need_ocr_list = []
    img_crop_list = []
    for page_ocr_res_list in ocr_res_list:
        for ocr_res in page_ocr_res_list:
            if "np_img" in ocr_res:
                need_ocr_list.append((page_ocr_res_list, ocr_res))
                img_crop_list.append(ocr_res.pop("np_img"))
    return need_ocr_list, img_crop_list


def _should_remove_low_confidence_ocr_text(ocr_text: str, ocr_score: float, ocr_res: dict[str, Any]) -> bool:
    """判断 OCR rec 结果是否应因低置信或竖排噪声被丢弃。"""
    if ocr_score < OcrConfidence.min_confidence:
        return True

    layout_res_bbox = ocr_res.get("bbox")
    if layout_res_bbox is None and ocr_res.get("poly") is not None:
        layout_res_bbox = [
            ocr_res["poly"][0],
            ocr_res["poly"][1],
            ocr_res["poly"][4],
            ocr_res["poly"][5],
        ]
    if layout_res_bbox is None:
        return True

    layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
    layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
    return (
        ocr_text
        in [
            "（204号",
            "（20",
            "（2",
            "（2号",
            "（20号",
            "号",
            "（204",
            "(cid:)",
            "(ci:)",
            "(cd:1)",
            "cd:)",
            "c)",
            "(cd:)",
            "c",
            "id:)",
            ":)",
            "√:)",
            "√i:)",
            "−i:)",
            "−:",
            "i:)",
        ]
        and ocr_score < 0.8
        and layout_res_width < layout_res_height
    )

def _apply_ocr_rec_results(
    local_model_context: HybridLocalModelContext,
    ocr_res_list: list[list[dict[str, Any]]],
) -> None:
    """执行 OCR rec 并把文本写回 sidecar，结果数量异常时显式报错。"""
    need_ocr_list, img_crop_list = _collect_ocr_rec_inputs(ocr_res_list)
    if not img_crop_list:
        return

    ocr_result_list = local_model_context.ocr_model.ocr(
        img_crop_list,
        det=False,
        tqdm_enable=True,
    )[0]

    if len(ocr_result_list) != len(need_ocr_list):
        raise ValueError(
            "Hybrid OCR rec result count mismatch: "
            f"ocr_result_list={len(ocr_result_list)}, need_ocr_list={len(need_ocr_list)}"
        )

    items_to_remove = []
    for index, (page_ocr_res_list, need_ocr_res) in enumerate(need_ocr_list):
        ocr_text, ocr_score = ocr_result_list[index]
        need_ocr_res["text"] = ocr_text
        need_ocr_res["score"] = float(f"{ocr_score:.3f}")
        if _should_remove_low_confidence_ocr_text(ocr_text, ocr_score, need_ocr_res):
            items_to_remove.append((page_ocr_res_list, need_ocr_res))

    for page_ocr_res_list, need_ocr_res in items_to_remove:
        if need_ocr_res in page_ocr_res_list:
            page_ocr_res_list.remove(need_ocr_res)


def _normalize_bbox_to_unit(item: dict[str, Any], page_width: int, page_height: int) -> bool:
    """将像素级bbox归一化为[0, 1]区间"""
    bbox = item.get("bbox")
    if bbox is None or len(bbox) != 4:
        return False

    x0, y0, x1, y1 = [float(v) for v in bbox]
    if 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0:
        normalized_bbox = [x0, y0, x1, y1]
    else:
        normalized_bbox = [
            x0 / page_width,
            y0 / page_height,
            x1 / page_width,
            y1 / page_height,
        ]
    item["bbox"] = [round(min(max(v, 0), 1), 3) for v in normalized_bbox]
    return True


def _normalize_bbox(
    inline_formula_list: list[list[dict[str, Any]]],
    ocr_res_list: list[list[dict[str, Any]]],
    images_pil_list: list[Image.Image],
) -> None:
    """归一化坐标并生成最终结果"""
    for page_inline_formula_list, page_ocr_res_list, page_pil_image in zip(inline_formula_list, ocr_res_list, images_pil_list):
        if page_inline_formula_list or page_ocr_res_list:
            page_width, page_height = page_pil_image.size
            # 处理公式列表
            for formula in page_inline_formula_list:
                _normalize_bbox_to_unit(formula, page_width, page_height)
            # 处理OCR结果列表
            for ocr_res in page_ocr_res_list:
                _normalize_bbox_to_unit(ocr_res, page_width, page_height)


def _build_inline_formula_model_item(formula: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "inline_formula",
        "bbox": list(formula["bbox"]),
        "latex": formula.get("latex", ""),
        "score": float(formula.get("score", 0.0)),
    }


def _build_ocr_text_model_item(ocr_res: dict[str, Any], keep_text: bool = True) -> dict[str, Any]:
    """构造 OCR det sidecar；VLM-OCR 路径可只保留空文本行提示。"""
    return {
        "type": "ocr_text",
        "bbox": list(ocr_res["bbox"]),
        "text": ocr_res.get("text", "") if keep_text else "",
        "score": float(ocr_res.get("score", 0.0)),
    }


def _merge_page_sidecar_items(
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    ocr_res_list: list[list[dict[str, Any]]],
    keep_ocr_text: bool = True,
) -> list[list[dict[str, Any]]]:
    merged_model_list: list[list[dict[str, Any]]] = []
    for page_model_list, page_inline_formula_list, page_ocr_res_list in zip(model_list, inline_formula_list, ocr_res_list):
        merged_page_model_list: list[dict[str, Any]] = list(page_model_list)
        merged_page_model_list.extend(
            _build_inline_formula_model_item(formula) for formula in page_inline_formula_list if formula.get("bbox") is not None
        )
        merged_page_model_list.extend(
            _build_ocr_text_model_item(ocr_res, keep_text=keep_ocr_text)
            for ocr_res in page_ocr_res_list
            if ocr_res.get("bbox") is not None
        )
        merged_model_list.append(merged_page_model_list)
    return merged_model_list


def _medium_bbox_to_quad(bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    """将普通 bbox 转为表格模型 OCR token 使用的四点框。"""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _normalize_medium_content(value: Any) -> str:
    """将 medium 本地模型输出的文本字段规范成 Hybrid block 可消费的字符串。"""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


def _apply_medium_formula_number_ocr(
    local_context: HybridLocalModelContext,
    model_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> None:
    """对 medium formula_number 裁剪图执行 OCR-rec，并把编号文本回填到原始 layout 项。"""
    need_rec_items: list[dict[str, Any]] = []
    formula_number_crops: list[np.ndarray] = []
    for block_list, np_img in zip(model_list, np_images):
        image_h, image_w = np_img.shape[:2]
        bgr_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        for block_item in block_list:
            if block_item.get("type") != BlockType.FORMULA_NUMBER:
                continue

            formula_number_bbox = normalize_to_int_bbox(
                _bbox_to_pixel_bbox(block_item.get("bbox"), (image_w, image_h)),
                image_size=(image_h, image_w),
            )
            if formula_number_bbox is None:
                continue

            # 使用 OCR rec 的标准旋转裁剪逻辑，保证 medium 编号裁图与正文 OCR-rec 输入一致。
            formula_number_crops.append(
                get_rotate_crop_image_for_text_rec(
                    bgr_image,
                    _medium_bbox_to_quad(formula_number_bbox).copy(),
                )
            )
            need_rec_items.append(block_item)

    if not formula_number_crops:
        return

    ocr_result_list = local_context.ocr_model.ocr(
        formula_number_crops,
        det=False,
        tqdm_enable=True,
        tqdm_desc="OCR-rec",
    )[0]
    if len(ocr_result_list) != len(need_rec_items):
        raise ValueError(
            "Hybrid medium formula number OCR rec result count mismatch: "
            f"ocr_result_list={len(ocr_result_list)}, need_rec_items={len(need_rec_items)}"
        )

    for block_item, ocr_result in zip(need_rec_items, ocr_result_list):
        if not ocr_result or len(ocr_result) < 2:
            continue
        ocr_text, _ = ocr_result
        normalized_text = _normalize_medium_content(ocr_text)
        if normalized_text:
            block_item["content"] = normalized_text


def _process_text_and_formulas(
    images_pil_list: list[Image.Image],
    model_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    effort: Literal["medium", "high", "xhigh"],
    local_model_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """处理OCR和公式识别"""

    # 遍历model_list,对文本块截图交由OCR识别
    # 根据 parse_mode 和 effort 决定需要ocr的文本块的类型以及只开det还是det+rec
    ocr_det_type, mfr_enable = _build_ocr_det_type_and_mfr_enable(
        parse_mode=parse_mode,
        effort=effort,
    )

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    mfd_res = _build_formula_inputs(images_layout_res)
    images_formula_list = mfd_res
    interline_enable = effort == "medium"

    # medium 识别行内和行间公式；high/xhigh 的 txt 路径只识别行内公式。
    if mfr_enable:
        images_formula_list = local_model_context.mfr_model.batch_predict(
            mfd_res,
            np_images,
            batch_size=BATCH_RATIO * MFR_BASE_BATCH_SIZE,
            interline_enable=interline_enable,
        )

    inline_formula_list, display_formula_list = _split_formula_results(images_formula_list)
    if effort == "medium":
        # 将行间公式span回填入block
        _apply_medium_display_formula_results(
            model_list,
            display_formula_list,
            images_pil_list,
        )
        # 使用ocr识别行间公式标号
        _apply_medium_formula_number_ocr(
            local_model_context,
            model_list,
            np_images,
        )

    # 行间公式标号回填到block
    for page_model_list in model_list:
        optimize_hybrid_formula_number_blocks(page_model_list)

    need_rec_img = parse_mode == "ocr" and effort == "medium"
    # vlm没有执行ocr，需要ocr_det
    ocr_res_list = _ocr_det(
        local_model_context,
        np_images,
        model_list,
        mfd_res,
        need_rec_img,
        ocr_det_type,
    )

    # 如果有rec_img则做ocr_rec
    if need_rec_img:
        _apply_ocr_rec_results(local_model_context, ocr_res_list)

    _normalize_bbox(inline_formula_list, ocr_res_list, images_pil_list)
    merged_model_list = _merge_page_sidecar_items(
        model_list,
        inline_formula_list,
        ocr_res_list,
    )
    return merged_model_list


def _collect_layout_doc_title_bboxes(layout_res: list[dict[str, Any]], page_size: tuple[int, int]) -> list[BBox]:
    """只收集layout小模型输出的doc_title框，忽略paragraph_title等其他类型。"""
    doc_title_bboxes: list[BBox] = []
    for layout_item in layout_res or []:
        if layout_item.get("label") != BlockType.DOC_TITLE:
            continue
        bbox = _bbox_to_pixel_bbox(layout_item.get("bbox"), page_size)
        if bbox is not None:
            doc_title_bboxes.append(bbox)
    return doc_title_bboxes


def _has_doc_title_overlap(title_bbox: BBox, doc_title_bboxes: list[BBox], overlap_threshold: float) -> bool:
    """判断VLM标题框是否与任一layout doc_title框达到最小框重叠阈值。"""
    return any(
        calculate_overlap_area_2_minbox_area_ratio(title_bbox, doc_title_bbox) >= overlap_threshold
        for doc_title_bbox in doc_title_bboxes
    )


def _apply_layout_title_split(
    model_list: list[list[dict[str, Any]]],
    images_layout_res: list[list[dict[str, Any]]],
    page_sizes: list[tuple[int, int]],
    overlap_threshold: float = LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD,
) -> None:
    """用layout doc_title框将VLM title拆分为doc_title和paragraph_title。"""
    for page_model_list, layout_res, page_size in zip(model_list, images_layout_res, page_sizes):
        doc_title_bboxes = _collect_layout_doc_title_bboxes(layout_res, page_size)
        for block in page_model_list:
            if block.get("type") != BlockType.TITLE:
                continue
            title_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if title_bbox is None:
                continue
            if _has_doc_title_overlap(title_bbox, doc_title_bboxes, overlap_threshold):
                block["type"] = BlockType.DOC_TITLE
            else:
                block["type"] = BlockType.PARAGRAPH_TITLE


def doc_analyze(
    pdf_bytes: bytes,
    effort: Literal["medium", "high", "xhigh"] = "high",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
) -> tuple[list[PageInfo], list[list[dict[str, Any]]]]:
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
                pdf_pages = _get_window_pdf_pages(pdf_doc, window)
                images_list = load_images_from_pdf_bytes_range(
                    pdf_bytes=pdf_bytes,
                    start_page_id=window.start,
                    end_page_id=window.end,
                    image_type="pil_img",
                )
                if len(pdf_pages) != len(images_list):
                    raise ValueError("Hybrid processing window PDF page count does not match image count")
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    _log_processing_window(window, page_count, len(images_pil_list))

                    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
                    images_layout_res = hybrid_model.layout_model.batch_predict(
                        images_pil_list,
                        batch_size=min(8, BATCH_RATIO * LAYOUT_BASE_BATCH_SIZE)
                    )

                    # 使用小模型layout时对layout的表格做旋转检测
                    if effort in ["medium", "high"]:
                        table_items = _collect_table_items(images_layout_res, np_images)
                        if table_items:
                            rotate_labels = hybrid_model.table_orientation_cls_model.batch_predict(
                                table_items,
                                det_batch_size=BATCH_RATIO * OCR_DET_BASE_BATCH_SIZE,
                                tqdm_enable=True,
                            )
                            _apply_table_rotate_labels(table_items, rotate_labels)

                    vl_style_layout_blocks = _build_vl_style_layout_blocks(images_layout_res, images_pil_list)

                    if parse_mode == "txt":
                        if effort == "medium":
                            window_model_list = vl_style_layout_blocks
                        elif effort == "high":
                            window_model_list = vlm_predictor.batch_extract_with_layout(
                                images=images_pil_list,
                                blocks_list=vl_style_layout_blocks,
                                not_extract_list=NOT_EXTRACT_TYPES,
                                image_analysis=False,
                            )
                        elif effort == "xhigh":
                            window_model_list = vlm_predictor.batch_two_step_extract(
                                images=images_pil_list,
                                not_extract_list=NOT_EXTRACT_TYPES,
                                image_analysis=image_analysis,
                            )
                            _apply_layout_title_split(
                                window_model_list,
                                images_layout_res,
                                [_normalize_page_size(image) for image in images_pil_list],
                            )
                        else:
                            raise ValueError(f"Unsupported analyze effort: {effort}")
                    elif parse_mode == "ocr":
                        if effort == "medium":
                            window_model_list = vl_style_layout_blocks
                        elif effort == "high":
                            window_model_list = vlm_predictor.batch_extract_with_layout(
                                images=images_pil_list,
                                blocks_list=vl_style_layout_blocks,
                                image_analysis=False,
                            )
                        elif effort == "xhigh":
                            window_model_list = vlm_predictor.batch_two_step_extract(
                                images=images_pil_list,
                                image_analysis=image_analysis,
                            )
                            _apply_layout_title_split(
                                window_model_list,
                                images_layout_res,
                                [_normalize_page_size(image) for image in images_pil_list],
                            )
                        else:
                            raise ValueError(f"Unsupported analyze effort: {effort}")
                    else:
                        raise ValueError(f"Unsupported parse mode: {parse_mode}")

                    window_model_list = _process_text_and_formulas(
                        images_pil_list,
                        window_model_list,
                        parse_mode,
                        effort,
                        hybrid_model,
                        images_layout_res,
                    )

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
    # pdf_path = "/Users/myhloli/pdf/截断合并/demo1-2.pdf"
    pdf_path = "/Users/myhloli/pdf/png/shubiao.png"
    pdf_bytes = read_fn(pdf_path)
    middle_json, model_list = doc_analyze(pdf_bytes, effort="medium", image_analysis=True)
    logger.info(f"middle_json: {middle_json}")
    logger.info(f"model_list: {model_list}")
