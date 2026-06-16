# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PPFormulaNetVisionConfig(PretrainedConfig):
    model_type = "pp_formulanet_vision"

    def __init__(
        self,
        image_size: int = 768,
        patch_size: int | list[int] | tuple[int, int] = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        output_channels: int = 256,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        initializer_range: float = 1e-10,
        qkv_bias: bool = True,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 14,
        global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11),
        mlp_dim: int = 3072,
        post_conv_in_channels: int = 256,
        post_conv_out_channels: int = 1024,
        post_conv_mid_channels: int = 512,
        decoder_hidden_size: int = 512,
        **kwargs,
    ):
        """保存 plus-L 视觉编码器参数，字段名对齐 safetensors config。"""
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = list(global_attn_indexes)
        self.mlp_dim = mlp_dim
        self.post_conv_in_channels = post_conv_in_channels
        self.post_conv_out_channels = post_conv_out_channels
        self.post_conv_mid_channels = post_conv_mid_channels
        self.decoder_hidden_size = decoder_hidden_size


class PPFormulaNetTextConfig(MBartConfig):
    model_type = "pp_formulanet_text"

    def __init__(
        self,
        vocab_size: int = 50000,
        max_position_embeddings: int = 2560,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 16,
        decoder_layers: int = 8,
        decoder_ffn_dim: int = 2048,
        decoder_attention_heads: int = 16,
        decoder_layerdrop: float = 0.0,
        use_cache: bool = True,
        is_encoder_decoder: bool = True,
        activation_function: str = "gelu",
        d_model: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        init_std: float = 0.02,
        scale_embedding: bool = True,
        pad_token_id: int | None = 1,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 2,
        decoder_start_token_id: int | None = 2,
        forced_eos_token_id: int | list[int] | None = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """用 MBartConfig 承载 PP-FormulaNet decoder，减少 transformers 4.57 适配面。"""
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            decoder_layerdrop=decoder_layerdrop,
            use_cache=use_cache,
            is_encoder_decoder=is_encoder_decoder,
            activation_function=activation_function,
            d_model=d_model,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            init_std=init_std,
            scale_embedding=scale_embedding,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.hidden_size = d_model


class PPFormulaNetConfig(PretrainedConfig):
    model_type = "pp_formulanet"
    is_composition = True

    def __init__(
        self,
        text_config: dict | PPFormulaNetTextConfig | None = None,
        vision_config: dict | PPFormulaNetVisionConfig | None = None,
        is_encoder_decoder: bool = True,
        **kwargs,
    ):
        """组合 text/vision 子配置，并把生成相关 token id 暴露到顶层 config。"""
        if isinstance(text_config, PPFormulaNetTextConfig):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            self.text_config = PPFormulaNetTextConfig(**text_config)
        else:
            logger.info("text_config is None. Initializing PPFormulaNetTextConfig.")
            self.text_config = PPFormulaNetTextConfig()

        if isinstance(vision_config, PPFormulaNetVisionConfig):
            self.vision_config = vision_config
        elif isinstance(vision_config, dict):
            self.vision_config = PPFormulaNetVisionConfig(**vision_config)
        else:
            logger.info("vision_config is None. Initializing PPFormulaNetVisionConfig.")
            self.vision_config = PPFormulaNetVisionConfig()

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            pad_token_id=self.text_config.pad_token_id,
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            decoder_start_token_id=self.text_config.decoder_start_token_id,
            forced_eos_token_id=self.text_config.forced_eos_token_id,
            tie_word_embeddings=self.text_config.tie_word_embeddings,
            **kwargs,
        )
        self.is_encoder_decoder = is_encoder_decoder

    def to_dict(self):
        """导出嵌套配置，便于 from_pretrained/save_pretrained 保持原始结构。"""
        output = super().to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.model_type
        return output
