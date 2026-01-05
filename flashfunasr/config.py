import os
from dataclasses import dataclass, field
from transformers import AutoConfig
import torch



@dataclass
class FunASRNanoConfig:
    architectures: list[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-06
    rope_scaling: dict | None = None
    rope_theta: float = 1000000.0
    sliding_window: int = None
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    transformers_version: str = "4.57.3"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
    # text_vocab_size: int = 151936
    # speech_vocab_size: int = 6562  # actually 6564, we only care about non-streaming inference, so cut off tokens (6562, 6563) that are only used for streaming TTS
    lm_head_bias: bool = True
    qkv_bias: bool = False
    fp16_flow: bool = True

@dataclass
class SamplingParams:
    temperature: float = 1.0
    min_tokens: int = 2
    max_tokens: int = 1024
    ignore_eos: bool = False
    top_k: int = 1
    # RasSampler parameters
    use_ras: bool = False
    win_size: int = 10
    tau_r: float = 0.1
    top_p: float = 0.8

@dataclass
class AudioEncoderConf:
    freeze: bool = True
    input_size: int = 560
    output_size: int = 512
    attention_heads: int =  4
    linear_units:int =  2048
    num_blocks:int =  50
    tp_blocks:int = 20
    dropout_rate: int = 0.1
    positional_dropout_rate:int = 0.1
    attention_dropout_rate:int =  0.1
    input_layer: str = 'pe'
    normalize_before: bool = True
    kernel_size: int = 11
    sanm_shfit:int = 0
    feat_permute: bool = True

@dataclass
class AudioAdaptorConf:
  downsample_rate: int = 1
  ffn_dim: int = 2048
  llm_dim: int = 1024
  encoder_dim: int = 512
  n_layer: int = 2
  freeze: bool = True

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 1536
    max_num_seqs: int = 8
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: FunASRNanoConfig | AutoConfig = field(default_factory=FunASRNanoConfig)
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    min_token_text_ratio: int = 2
    max_token_text_ratio: int = 20
    rank: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        max_pos = getattr(self.hf_config, "max_position_embeddings", 4096)
        self.max_model_len = min(self.max_model_len, max_pos)
        assert self.max_num_batched_tokens >= self.max_model_len



