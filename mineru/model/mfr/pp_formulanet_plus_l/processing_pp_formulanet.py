# Copyright (c) Opendatalab. All rights reserved.
import re

from transformers import AutoTokenizer
from transformers.image_processing_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging

from .image_processing_pp_formulanet import PPFormulaNetImageProcessor


logger = logging.get_logger(__name__)


class PPFormulaNetProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "PPFormulaNetImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        """组合 image processor 与 tokenizer，保持 transformers ProcessorMixin 兼容接口。"""
        # 4.57 的 ProcessorMixin 会强制到 transformers 全局注册表查找自定义类，这里直接保存本地对象。
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self._text_reg = re.compile(r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})")
        self._macro_pattern = re.compile(r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})")
        self._protected_macros = {"\\operatorname", "\\mathrm", "\\text", "\\mathbf"}
        letter = r"[a-zA-Z]"
        noletter = r"[\W_^\d]"
        self._rule_noletter_noletter = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter))
        self._rule_noletter_letter = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter))
        self._rule_letter_noletter = re.compile(r"(%s)\s+?(%s)" % (letter, noletter))
        self._chinese_text_wrapping_pattern = re.compile(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从本地 plus-L snapshot 构造 processor；不依赖 AutoProcessor 注册。"""
        image_processor = PPFormulaNetImageProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(self, images, **kwargs):
        """只处理图像输入，返回可直接传给模型的 pixel_values。"""
        image_inputs = self.image_processor(images, **kwargs)
        return BatchFeature({**image_inputs})

    def batch_decode(self, *args, **kwargs):
        """代理 tokenizer 解码，便于测试和运行时统一后处理入口。"""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def post_process_generation(self, text: str) -> str:
        """清理生成的 LaTeX 文本，逻辑对齐 transformers 5.12 的 PPFormulaNetProcessor。"""
        text = self.remove_chinese_text_wrapping(text)
        try:
            from ftfy import fix_text

            text = fix_text(text)
        except ImportError:
            logger.warning_once("ftfy is not installed, skipping PP-FormulaNet text normalization.")
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """移除 LaTeX 中多余空格，同时保护 operatorname/text 等宏。"""
        names = []
        for x in self._text_reg.findall(text):
            matches = self._macro_pattern.findall(x[0])
            for m in matches:
                if m not in self._protected_macros and m.strip() != "":
                    text = text.replace(m, m + "XXXXXXX")
                    text = text.replace(" ", "")
                    names.append(text)

        if names:
            text = self._text_reg.sub(lambda match: str(names.pop(0)), text)

        new_text = text
        while True:
            text = new_text
            new_text = self._rule_noletter_noletter.sub(r"\1\2", text)
            new_text = self._rule_noletter_letter.sub(r"\1\2", new_text)
            new_text = self._rule_letter_noletter.sub(r"\1\2", new_text)
            if new_text == text:
                break
        return new_text.replace("XXXXXXX", " ")

    def remove_chinese_text_wrapping(self, formula: str) -> str:
        """去掉中文公式外层的 \\text{} 包裹，避免 Markdown 中重复文本模式。"""
        return self._chinese_text_wrapping_pattern.sub(
            lambda match: match.group(1),
            formula,
        ).replace('"', "")

    def post_process(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """批量解码并执行公式后处理。"""
        generated_texts = self.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )
        return [self.post_process_generation(text) for text in generated_texts]
