# Copyright (c) Opendatalab. All rights reserved.
import os

import torch
from loguru import logger
from tqdm import tqdm

from mineru.model.mfr.pp_formulanet_plus_m.predict_formula import (
    FormulaRecognizer as FormulaRecognizerPlusM,
)

from ..utils import build_mfr_batch_groups
from .modeling_pp_formulanet import PPFormulaNetForConditionalGeneration
from .processing_pp_formulanet import PPFormulaNetProcessor


class FormulaRecognizerPlusL:
    dtype_env_var = "MINERU_FORMULA_PLUS_L_DTYPE"

    def __init__(
        self,
        weight_dir,
        device="cpu",
    ):
        """加载 PP-FormulaNet_plus-L safetensors 模型和 processor。"""
        self.weight_dir = os.fspath(weight_dir)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.inference_dtype = self._resolve_inference_dtype(self.device)
        self.processor = PPFormulaNetProcessor.from_pretrained(self.weight_dir)
        self.model = PPFormulaNetForConditionalGeneration.from_pretrained(
            self.weight_dir,
            dtype=self.inference_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def _supports_bf16(cls, device: torch.device) -> bool:
        """判断当前设备是否适合使用 bf16 推理。"""
        if device.type == "cuda":
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        elif device.type == "mps":
            return True
        else:
            return False

    @classmethod
    def _resolve_inference_dtype(cls, device: torch.device) -> torch.dtype:
        """从环境变量解析 plus-L 推理精度，默认使用 bf16。"""
        dtype_name = os.getenv(cls.dtype_env_var, "bf16").strip().lower()
        if dtype_name in {"fp32", "float32"}:
            return torch.float32
        if dtype_name == "auto":
            return torch.bfloat16 if cls._supports_bf16(device) else torch.float32
        if dtype_name not in {"bf16", "bfloat16"}:
            logger.warning(
                f"Invalid {cls.dtype_env_var} value: {dtype_name}, fallback to bf16."
            )
        if cls._supports_bf16(device):
            return torch.bfloat16
        logger.warning(
            f"{cls.dtype_env_var}=bf16 is not supported on device {device}, fallback to fp32."
        )
        return torch.float32

    def _move_inputs_to_device(self, inputs):
        """将 processor 输出移动到目标设备，并把图像张量转换为推理精度。"""
        moved_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved_inputs[key] = value
                continue
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                moved_inputs[key] = value.to(
                    device=self.device,
                    dtype=self.inference_dtype,
                )
            else:
                moved_inputs[key] = value.to(self.device)
        return moved_inputs

    @staticmethod
    def _normalize_bbox(bbox, image):
        """复用 plus-M 的 bbox 取整、越界裁剪和非法框过滤逻辑。"""
        return FormulaRecognizerPlusM._normalize_bbox(bbox, image)

    @staticmethod
    def _item_to_bbox(item, image):
        """从公式检测结果中提取并规范化 bbox。"""
        return FormulaRecognizerPlusL._normalize_bbox(item.get("bbox"), image)

    def _build_formula_items(self, mfd_res, image, interline_enable=True):
        """筛选需要识别的公式项，并记录可裁剪的候选框。"""
        formula_list = []
        crop_targets = []

        for item in mfd_res or []:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label not in ["inline_formula", "display_formula"]:
                continue
            if not interline_enable and label == "display_formula":
                continue

            new_item = dict(item)
            new_item.setdefault("latex", "")
            formula_list.append(new_item)

            bbox = self._item_to_bbox(new_item, image)
            if bbox is not None:
                crop_targets.append((new_item, bbox))

        return formula_list, crop_targets

    def _predict_sorted_images(self, sorted_images, batch_groups):
        """按面积分组后的顺序批量生成公式 LaTeX。"""
        rec_formula = []
        if not sorted_images:
            return rec_formula

        with torch.inference_mode():
            with tqdm(total=len(sorted_images), desc="MFR Predict") as pbar:
                for batch_group in batch_groups:
                    batch_images = [sorted_images[i] for i in batch_group]
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    inputs = self._move_inputs_to_device(inputs)
                    generated_outputs = self.model.generate(**inputs)
                    rec_formula.extend(self.processor.post_process(generated_outputs))
                    pbar.update(len(batch_group))
        return rec_formula

    def predict(
        self,
        mfd_res,
        image,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        """保持现有 MFR 单图 predict 接口。"""
        return self.batch_predict(
            [mfd_res],
            [image],
            batch_size=batch_size,
            interline_enable=interline_enable,
        )[0]

    def batch_predict(
        self,
        images_mfd_res: list,
        images: list,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        """保持现有 MFR batch_predict 行为，按面积排序推理后恢复原顺序回填 latex。"""
        if not images_mfd_res:
            return []

        if len(images_mfd_res) != len(images):
            raise ValueError("images_mfd_res and images must have the same length.")

        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []

        for mfd_res, image in zip(images_mfd_res, images):
            formula_list, crop_targets = self._build_formula_items(
                mfd_res,
                image,
                interline_enable=interline_enable,
            )

            for formula_item, (xmin, ymin, xmax, ymax) in crop_targets:
                bbox_img = image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)
                backfill_list.append(formula_item)

            images_formula_list.append(formula_list)

        if not image_info:
            return images_formula_list

        image_info.sort(key=lambda x: x[0])
        sorted_areas = [x[0] for x in image_info]
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]
        index_mapping = {
            new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)
        }

        formula_requested_batch_size = max(1, batch_size // 2)
        batch_groups = build_mfr_batch_groups(
            sorted_areas,
            formula_requested_batch_size,
        )

        rec_formula = self._predict_sorted_images(sorted_images, batch_groups)

        unsorted_results = [""] * len(rec_formula)
        for new_idx, latex in enumerate(rec_formula):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list
