import os
from typing import Optional, Mapping, Any
import torch
from  torch import nn

from flashfunasr.modules.audio_encoder.attention import MultiHeadedAttentionSANM
from flashfunasr.modules.audio_encoder.layer import SinusoidalPositionEncoder, PositionwiseFeedForward, \
    EncoderLayerSANM, LayerNorm

def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)

class SenseVoiceEncoderSmall(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = SinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(output_size)

        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        maxlen = xs_pad.shape[1]
        masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]

        xs_pad *= self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        # forward encoder2
        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens

    def load_weights(self, config, device):
        checkpoint_path = os.path.join(config.model, "model.pt")

        print(f"Loading weights from {checkpoint_path}...")

        # 1. 加载 Checkpoint
        # map_location 确保权重被加载到指定的设备 (cpu/cuda)

        loaded_state_dict = torch.load(checkpoint_path, map_location=device)
        # 兼容有些 checkpoint 保存为 {"state_dict": ...} 的格式，也兼容直接保存 state_dict 的格式
        if "state_dict" in loaded_state_dict:
            loaded_state_dict = loaded_state_dict["state_dict"]


        # 获取当前模型的 state_dict 用于对比
        model_state_dict = self.state_dict()
        new_state_dict = {}

        # 2. Key 转换与清洗
        for k, v in loaded_state_dict.items():
            # 核心逻辑：移除 'audio_encoder.' 前缀
            if k.startswith("audio_encoder."):
                # 去掉前14个字符 ("audio_encoder.")
                new_key = k[14:]
            else:
                # 如果没有前缀，保持原样（防止有不带前缀的权重）
                new_key = k

            # 3. 匹配与赋值
            if new_key in model_state_dict:
                target_shape = model_state_dict[new_key].shape

                # 形状检查
                if v.shape != target_shape:
                    print(
                        f"[Warning] Shape mismatch for {new_key}: Checkpoint {v.shape} != Model {target_shape}. Skipping.")
                    continue

                new_state_dict[new_key] = v
            else:
                # 这里的 verbose 可以关掉，防止打印太多无关的 keys
                # print(f"[Info] Key {new_key} in checkpoint but not in model. Ignored.")
                pass

        # 4. 加载到模型中
        # strict=False 是必须的，因为：
        # (1) Checkpoint 可能包含优化器状态或其他无关参数
        # (2) 你的模型结构里有 'tp_encoders'，但提供的 keys 列表里没有看到对应的权重。
        #     如果 checkpoint 里确实没有 tp_encoders，strict=True 会报错。
        keys_missing, keys_unexpected = self.load_state_dict(new_state_dict, strict=False)

        # 5. 打印加载报告
        print("-" * 50)
        print("Load Report:")

        # 过滤掉不需要关注的 missing keys (比如 positional encodings 如果是固定的)
        real_missing = [k for k in keys_missing if "embed" not in k]

        if len(real_missing) > 0:
            print(f"WARNING: The following layers were NOT loaded (Total {len(real_missing)}):")
            # 只打印前5个，避免刷屏
            for k in real_missing[:5]:
                print(f"  - {k}")
            if len(real_missing) > 5:
                print(f"  ... and {len(real_missing) - 5} more.")

            # 特别检查 tp_encoders
            if any("tp_encoders" in k for k in real_missing):
                print(
                    "\n[Note] 'tp_encoders' are missing. If this is a streaming model, ensure your checkpoint supports it.")
        else:
            print("All model keys successfully loaded!")

        print("-" * 50)

