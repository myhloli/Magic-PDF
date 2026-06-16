# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mbart.modeling_mbart import MBartDecoder, shift_tokens_right
from transformers.utils import ModelOutput, logging

from .configuration_pp_formulanet import PPFormulaNetConfig, PPFormulaNetVisionConfig


logger = logging.get_logger(__name__)


class PPFormulaNetPreTrainedModel(PreTrainedModel):
    config_class = PPFormulaNetConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = False

    def _init_weights(self, module):
        """初始化本地 PP-FormulaNet 模块，保持权重加载前的随机模型可用于轻量测试。"""
        std = getattr(self.config, "initializer_range", None)
        if std is None:
            text_config = getattr(self.config, "text_config", None)
            vision_config = getattr(self.config, "vision_config", None)
            std = getattr(text_config, "init_std", getattr(vision_config, "initializer_range", 0.02))

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PPFormulaNetVisionModel):
            if module.pos_embed is not None:
                nn.init.constant_(module.pos_embed, 0.0)
        elif isinstance(module, PPFormulaNetVisionAttention) and module.use_rel_pos:
            nn.init.constant_(module.rel_pos_h, 0.0)
            nn.init.constant_(module.rel_pos_w, 0.0)


@dataclass
class PPFormulaNetVisionEncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class PPFormulaNetVisionAttention(nn.Module):
    def __init__(self, config: PPFormulaNetVisionConfig, window_size: int):
        """构建带相对位置编码的视觉 attention，命名对齐 HF safetensors。"""
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )
        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """按 query/key 尺寸插值相对位置编码。"""
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).transpose(1, 2),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos_resized[relative_coords.long()]

    def get_decomposed_rel_pos(
        self,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor:
        """计算 H/W 分解相对位置偏置。"""
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)
        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        return rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """执行窗口内或全局 self-attention。"""
        batch_size, height, width, _ = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)
        attn_weights = (query * self.scale) @ key.transpose(-2, -1)
        if self.use_rel_pos:
            decomposed_rel_pos = self.get_decomposed_rel_pos(
                query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )
            attn_weights = attn_weights + decomposed_rel_pos.reshape_as(attn_weights)
        attn_weights = torch.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        return self.proj(attn_output), attn_weights


class PPFormulaNetMultiModalProjector(nn.Module):
    def __init__(self, config: PPFormulaNetVisionConfig):
        """将视觉特征投影到 decoder hidden size。"""
        super().__init__()
        self.conv1 = nn.Conv2d(
            config.post_conv_in_channels,
            config.post_conv_mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            config.post_conv_mid_channels,
            config.post_conv_out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.linear_1 = nn.Linear(config.post_conv_out_channels, config.post_conv_out_channels)
        self.linear_2 = nn.Linear(config.post_conv_out_channels, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        """输出供文本 decoder cross-attention 使用的序列特征。"""
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.linear_1(hidden_states)
        return self.linear_2(hidden_states)


class PPFormulaNetMLPBlock(nn.Module):
    def __init__(self, config: PPFormulaNetVisionConfig):
        """视觉 Transformer 的 MLP block。"""
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行两层前馈网络并返回视觉特征增量。"""
        return self.lin2(self.act(self.lin1(hidden_states)))


class PPFormulaNetVisionLayer(nn.Module):
    def __init__(self, config: PPFormulaNetVisionConfig, window_size: int):
        """单层视觉 encoder，支持局部窗口和指定层全局 attention。"""
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = PPFormulaNetVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = PPFormulaNetMLPBlock(config)
        self.window_size = window_size

    def window_partition(self, hidden_states: torch.Tensor, window_size: int):
        """把特征切分为不重叠窗口，不足处补齐。"""
        batch_size, height, width, channel = hidden_states.shape
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w
        hidden_states = hidden_states.reshape(
            batch_size,
            pad_height // window_size,
            window_size,
            pad_width // window_size,
            window_size,
            channel,
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
            -1, window_size, window_size, channel
        )
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self,
        windows: torch.Tensor,
        window_size: int,
        padding_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        """把窗口结果还原成原始空间尺寸。"""
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size,
            pad_height // window_size,
            pad_width // window_size,
            window_size,
            window_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
            batch_size, pad_height, pad_width, -1
        )
        return hidden_states[:, :height, :width, :].contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行视觉 Transformer 层的归一化、attention 和 MLP 残差计算。"""
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)
        hidden_states, _ = self.attn(hidden_states)
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class PPFormulaNetPatchEmbeddings(nn.Module):
    def __init__(self, config: PPFormulaNetVisionConfig):
        """将输入图像切 patch 并投影到 hidden size。"""
        super().__init__()
        image_size = config.image_size if isinstance(config.image_size, collections.abc.Iterable) else (
            config.image_size,
            config.image_size,
        )
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (
            config.patch_size,
            config.patch_size,
        )
        self.image_size = tuple(image_size)
        self.patch_size = tuple(patch_size)
        self.num_channels = config.num_channels
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor):
        """校验图像尺寸后生成 patch embedding。"""
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError("pixel_values channel dimension does not match PPFormulaNetVisionConfig.")
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model "
                f"({self.image_size[0]}*{self.image_size[1]})."
            )
        return self.projection(pixel_values).permute(0, 2, 3, 1)


class PPFormulaNetLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, *, eps=1e-6, data_format="channels_last", **kwargs):
        """支持 channels_first 的 LayerNorm，命名与 PP-FormulaNet 视觉 neck 对齐。"""
        super().__init__(normalized_shape, eps=eps, **kwargs)
        self.data_format = data_format

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """按 data_format 对 channels_first/channels_last 特征做 LayerNorm。"""
        if self.data_format == "channels_first":
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            return features.permute(0, 3, 1, 2)
        return super().forward(features)


class PPFormulaNetVisionNeck(nn.Module):
    def __init__(self, config: PPFormulaNetVisionConfig):
        """视觉 encoder 后处理 neck，输出 projector 所需通道数。"""
        super().__init__()
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = PPFormulaNetLayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = PPFormulaNetLayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        """把 channels-last 视觉特征转换为 neck 输出特征图。"""
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.layer_norm1(self.conv1(hidden_states))
        return self.layer_norm2(self.conv2(hidden_states))


class PPFormulaNetVisionModel(PPFormulaNetPreTrainedModel):
    config_class = PPFormulaNetVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: PPFormulaNetVisionConfig):
        """构建 PP-FormulaNet plus-L 视觉 encoder。"""
        super().__init__(config)
        self.config = config
        self.image_size = config.image_size
        self.patch_embed = PPFormulaNetPatchEmbeddings(config)
        self.pos_embed = None
        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            window_size = config.window_size if i not in config.global_attn_indexes else 0
            self.layers.append(PPFormulaNetVisionLayer(config, window_size=window_size))
        self.neck = PPFormulaNetVisionNeck(config)
        self.multi_modal_projector = PPFormulaNetMultiModalProjector(config)
        self.post_init()

    def get_input_embeddings(self):
        """返回 patch embedding 模块，兼容 PreTrainedModel 接口。"""
        return self.patch_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """编码公式图片，pooler_output 作为 decoder cross-attention 输入。"""
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
        all_hidden_states = () if output_hidden_states else None
        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer_module(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        hidden_states = self.neck(hidden_states)
        pooler_output = self.multi_modal_projector(hidden_states)
        if not return_dict:
            return (hidden_states, pooler_output, all_hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class PPFormulaNetModel(PPFormulaNetPreTrainedModel):
    def __init__(self, config: PPFormulaNetConfig):
        """组合 plus-L vision encoder 与 MBart decoder。"""
        super().__init__(config)
        self.decoder = MBartDecoder(config.text_config)
        self.encoder = PPFormulaNetVisionModel(config.vision_config)
        self.post_init()

    def get_encoder(self):
        """返回视觉 encoder，供 generate 预编码图像使用。"""
        return self.encoder

    def get_decoder(self):
        """返回文本 decoder，供生成框架访问。"""
        return self.decoder

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        encoder_outputs: Union[tuple, BaseModelOutputWithPooling, None] = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        """执行 encoder-decoder 前向，兼容 generate 传入的 encoder_outputs。"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError("decoder_input_ids or input_ids must be specified for decoder forward.")
            decoder_input_ids = shift_tokens_right(
                input_ids,
                self.config.text_config.pad_token_id,
                self.config.text_config.decoder_start_token_id,
            )

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You must specify pixel_values when encoder_outputs is None.")
            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )
        elif not isinstance(encoder_outputs, BaseModelOutputWithPooling):
            encoder_outputs = BaseModelOutputWithPooling(
                last_hidden_state=encoder_outputs[0],
                pooler_output=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.pooler_output,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        if not return_dict:
            return decoder_outputs + (encoder_outputs.last_hidden_state,)
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class PPFormulaNetForConditionalGeneration(PPFormulaNetPreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config: PPFormulaNetConfig):
        """用于公式识别的条件生成模型，输出 tokenizer token logits。"""
        super().__init__(config)
        self.model = PPFormulaNetModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        """返回底层视觉 encoder。"""
        return self.model.get_encoder()

    def get_decoder(self):
        """返回底层文本 decoder。"""
        return self.model.get_decoder()

    def get_output_embeddings(self) -> nn.Module:
        """返回语言模型输出层。"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """替换语言模型输出层，兼容 transformers 保存/加载接口。"""
        self.lm_head = new_embeddings

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: Union[tuple, BaseModelOutputWithPooling, None] = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        """执行条件生成前向；训练时可传 labels 计算交叉熵。"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels,
                self.config.text_config.pad_token_id,
                self.config.text_config.decoder_start_token_id,
            )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.config.text_config.vocab_size), labels.reshape(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        encoder_outputs=None,
        cache_position=None,
        **kwargs,
    ):
        """适配 transformers 4.57 generate：后续步只传 decoder token 与缓存。"""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
            "cache_position": cache_position,
        }
