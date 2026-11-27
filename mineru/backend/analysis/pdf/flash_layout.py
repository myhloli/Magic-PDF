# Copyright (c) Opendatalab. All rights reserved.
"""为 Flash/TXT 的告警页批量运行布局模型，按请求内物理页号提交完整结果。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

from docvortex.document.pdf.visuals import attach_visual_block_images
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


def replace_flash_layout_pages(
    document: PDFDocument, native_pages: list[list[dict]], page_indices: list[int], batch_size: int
) -> set[int]:
    """只渲染命中的物理页；回填及素材裁切全部成功后才替换该页的原生块。"""
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
                        attach_visual_block_images([blocks], [image], page_start_index=index)
                    except Exception as exc:
                        logger.warning("Flash TXT kept native layout for page {} after refill failure: {}", index + 1, exc)
                        continue
                    native_pages[index] = blocks
                    completed.add(index)
    finally:
        release_document(device, clean_memory)
    return completed
