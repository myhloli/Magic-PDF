# Copyright (c) Opendatalab. All rights reserved.
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from loguru import logger
from mineru.backend.local_model_runtime import AtomModelSingleton, HybridLocalModelContextSingleton
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.pdf_image_tools import load_images_from_pdf_bytes_range

from ...utils.image_payload import ImagePayloadCache
from ...utils.pdf_document import PDFDocument
from ...types import PageInfo
from ...utils.config_reader import get_processing_window_size


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 让mps可以fallback

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8


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


def doc_analyze(
    pdf_bytes: bytes,
    effort: Literal["medium", "high", "xhigh"] = "high",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    image_cache: ImagePayloadCache | None = None,
) -> tuple[list[PageInfo], list[list[dict[str, Any]]], bool]:
    batch_ratio = 2
    pdf_doc = PDFDocument(pdf_bytes)
    parse_mode = pdf_doc.classify()

    page_count = pdf_doc.page_count
    configured_window_size = get_processing_window_size(default=64)
    windows = _build_processing_windows(page_count, configured_window_size)
    _log_processing_window_plan(page_count, configured_window_size, len(windows))

    infer_start = time.time()
    progress_bar = None
    last_append_end_time = None

    hybrid_model_singleton = HybridLocalModelContextSingleton()
    hybrid_model = hybrid_model_singleton.get_model()
    atom_model_manager = AtomModelSingleton()

    try:
        for window in windows:
            images_list = load_images_from_pdf_bytes_range(
                pdf_bytes=pdf_bytes,
                start_page_id=window.start,
                end_page_id=window.end,
                image_type="pil_img",
            )
            images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
            _log_processing_window(window, page_count, len(images_pil_list))

            np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
            images_layout_res = hybrid_model.layout_model.batch_predict(
                images_pil_list,
                batch_size=min(8, batch_ratio * LAYOUT_BASE_BATCH_SIZE)
            )

            if effort == "medium":
                pass
            else:
                vlm_runtime = _load_vlm_runtime()
                vlm_backend = get_vlm_engine(inference_engine="auto", is_async=False)
                vlm_predictor = vlm_runtime["ModelSingleton"]().get_model(backend=vlm_backend)
                if effort == "high":
                    pass
                elif effort == "xhigh":
                    pass

