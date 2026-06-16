# Copyright (c) Opendatalab. All rights reserved.
import json
import os

from transformers.models.nougat.image_processing_nougat import NougatImageProcessor


class PPFormulaNetImageProcessor(NougatImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_crop_margin: bool = True,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        do_pad: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        **kwargs,
    ):
        """初始化 plus-L 图像预处理，默认参数对齐 HF processor_config.json。"""
        super().__init__(
            do_crop_margin=do_crop_margin,
            do_resize=do_resize,
            size=size or {"height": 768, "width": 768},
            do_thumbnail=do_thumbnail,
            do_align_long_axis=do_align_long_axis,
            do_pad=do_pad,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean or [0.7931, 0.7931, 0.7931],
            image_std=image_std or [0.1738, 0.1738, 0.1738],
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从 plus-L 独立仓库的 processor_config.json 读取图像预处理配置。"""
        config_path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        image_config = {}
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                processor_config = json.load(f)
            image_config = processor_config.get("image_processor", {})
        image_config.pop("image_processor_type", None)
        image_config.pop("return_tensors", None)
        image_config.update(kwargs)
        return cls(**image_config)
