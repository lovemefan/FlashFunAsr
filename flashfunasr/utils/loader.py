import dataclasses
import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn
import numpy as np
from flashfunasr.config import FunASRNanoConfig


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


import torch
import torch.nn as nn


def load_llm_model_weights(model: torch.nn.Module, checkpoint_path: str, device: str = "cuda"):
    """
    加载 FlashFunASR 的 Qwen3 LLM 权重，支持 MergedColumnParallelLinear (QKV/GateUp 融合)。
    """
    print(f"Loading weights from {checkpoint_path}...")

    # 1. 定义映射规则 (直接硬编码或从配置读取)
    # 格式: 源模块名 -> (目标模块名, Shard_ID)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    model_state_dict = model.state_dict()
    loaded_keys = set()

    # 加载 Checkpoint
    try:
        state_dict = torch.load(f"{checkpoint_path}/model.pt", map_location="cpu")["state_dict"]
    except Exception as e:
        print(f"Error loading file: {e}")
        return model

    # 2. 遍历 Checkpoint 中的每一个权重
    for key, param in state_dict.items():
        # A. 路径清洗: 移除 'llm.' 前缀
        if key.startswith("llm."):
            clean_key = key[4:]
        else:
            clean_key = key

        # B. 解析 Key 结构
        # 假设 key 是 "model.layers.0.self_attn.q_proj.weight"
        # splitting 后: parent_path="model.layers.0.self_attn", leaf_module="q_proj", param_type="weight"
        parts = clean_key.rsplit(".", 2)
        if len(parts) < 3:
            # 处理非 layer 层的权重 (如 embedding, norm 等)
            parent_path, leaf_module, param_type = "", parts[-2], parts[-1]
            full_param_name = clean_key
        else:
            parent_path, leaf_module, param_type = parts[0], parts[1], parts[2]
            full_param_name = clean_key

        param = param.to(device)

        # C. 判断是否属于 "融合层" (Merged Layer)
        if leaf_module in packed_modules_mapping:
            # === 处理融合权重 (QKV, GateUp) ===
            target_module_name, shard_id_raw = packed_modules_mapping[leaf_module]

            # 构造目标模块在 model 中的完整路径
            # 例如: "model.layers.0.self_attn.qkv_proj"
            target_module_path = f"{parent_path}.{target_module_name}" if parent_path else target_module_name

            try:
                # 获取模型中实际的 Layer 对象
                submodule = model.get_submodule(target_module_path)

                # 检查该层是否有 weight_loader 方法 (这是 nano-vllm 的核心)
                if hasattr(submodule, "weight_loader"):
                    # 调用自定义加载器：它会自动处理切片和拷贝
                    # 注意：我们要传 param (checkpoints里的权重) 进去
                    if param_type == "weight":
                        submodule.weight_loader(submodule.weight, param, loaded_shard_id=shard_id_raw)
                        loaded_keys.add(full_param_name)  # 记录原始 key 已处理

                        # 这是一个特殊的“虚拟 Key”添加，用于后续完整性检查
                        # 因为 model_state_dict 里只有 qkv_proj.weight，没有 q_proj.weight
                        # 我们把 qkv_proj.weight 标记为已加载的一部分
                        loaded_keys.add(f"{target_module_path}.weight")
                    else:
                        # Bias 通常比较简单，或者 Merged 层不支持 bias
                        pass
                else:
                    print(f"Warning: {target_module_path} is mapped but has no weight_loader!")

            except AttributeError:
                print(f"Error: Could not find submodule {target_module_path} in model.")

        else:
            # === 处理普通权重 (Embedding, Norm, DownProj, OProj) ===
            # DownProj 和 OProj 虽然不需要融合，但如果是 RowParallelLinear，
            # 它们通常也需要 weight_loader 来处理 TP 切分。
            # 如果你的 down_proj 是 RowParallelLinear，建议也用 weight_loader。

            # 简单起见，这里先尝试用 get_submodule 查找是否有 weight_loader
            # 如果没有，回退到原来的直接赋值

            target_module_path = f"{parent_path}.{leaf_module}" if parent_path else leaf_module
            try:
                submodule = model.get_submodule(target_module_path)
                if hasattr(submodule, "weight_loader") and param_type == "weight":
                    # 对于 RowParallel (如 down_proj)，不需要 shard_id，或者 shard_id 默认为 0
                    # 但通常 RowParallelLinear 不需要 shard_id 参数，或者内部忽略它
                    # 这里需要根据你的 RowParallelLinear 实现调整
                    try:
                        submodule.weight_loader(submodule.weight, param)
                    except TypeError:
                        # 如果 weight_loader 需要 shard_id 但我们没有 (比如 down_proj)
                        # 传入 0 试试
                        submodule.weight_loader(submodule.weight, param, loaded_shard_id=0)

                    loaded_keys.add(clean_key)
                    continue
            except:
                pass

            # 标准直接赋值回退 (Fallback)
            if clean_key in model_state_dict:
                target_shape = model_state_dict[clean_key].shape
                if param.shape != target_shape:
                    if param.T.shape == target_shape:
                        param = param.T
                        print(f"Transposed {clean_key}")
                    else:
                        # 忽略 shape 不匹配，可能是 TP 导致显存里是切分后的
                        # 如果这里报错，说明上面应该用 weight_loader 而没用到
                        print(f"Skipping {clean_key}: Shape mismatch {param.shape} vs {target_shape} (TP enabled?)")
                        continue

                with torch.no_grad():
                    model_state_dict[clean_key].copy_(param)
                    loaded_keys.add(clean_key)

    print("Weights loaded successfully.")
    return model

def load_text_llm(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    tmp_weights = torch.load(f"{path}/model.pt", map_location="cpu", weights_only=True)["state_dict"]

    for weight_name in tmp_weights.keys():
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = param.weight_loader
                weight_loader(param, tmp_weights.get_tensor(weight_name), shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tmp_weights.get_tensor(weight_name))


def load_speech_llm(model: nn.Module, path: str, hf_config: FunASRNanoConfig):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # NOTE(xcsong): 1. load speech embedding + sos/taskid embedding + lm head
    embedding_weights = {}
    tmp_weights = torch.load(f"{path}/model.pt", map_location="cpu", weights_only=True)["state_dict"]
    missed, missed_names = 0, []
    for k, v in tmp_weights.items():
        if k == "speech_embedding.weight":  # torch.Size([6564, 896])
            speech_embedding_size = hf_config.speech_vocab_size  # 6562
            # NOTE(xcsong): padding to 6592 for vllm tensor parallel
            if speech_embedding_size != v.shape[0]:  # [6564, 896] -> [6562, 896]
                assert speech_embedding_size <= v.shape[0], f"speech_embedding_size should be less than or equal to {v.shape[0]}, but got {speech_embedding_size}"
                v = v[:speech_embedding_size, :]
            embedding_weights["speech_embedding.weight"] = v
        elif k == "llm_embedding.weight":  # torch.Size([2, 896]), eos and task_id
            assert v.shape[0] == 2, f"llm_embedding.weight should be of shape [2, 896], but got {v.shape}"
            embedding_weights["llm_embedding.weight"] = v
        elif k == "llm.model.model.embed_tokens.weight":  # torch.Size([151936, 896])
            embedding_weights["model.embed_tokens.weight"] = v
        elif k == "llm_decoder.weight":  # torch.Size([6564, 896])
            lm_head_size = hf_config.speech_vocab_size  # 6562
            if lm_head_size != v.shape[0]:  # [6564, 896] -> [6562, 896]
                assert lm_head_size <= v.shape[0], f"lm_head_size should be less than or equal to {v.shape[0]}, but got {lm_head_size}"
                v = v[:lm_head_size, :]
            param = model.get_parameter("lm_head.weight")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, v)
        elif k == "llm_decoder.bias":  # torch.Size([6564])
            lm_head_size = hf_config.speech_vocab_size  # 6562
            if lm_head_size != v.shape[0]:  # [6564] -> [6562]
                assert lm_head_size <= v.shape[0], f"lm_head_size should be less than or equal to {v.shape[0]}, but got {lm_head_size}"
                v = v[:lm_head_size]
            param = model.get_parameter("lm_head.bias")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, v)
        elif "llm.model." in k:
            weight_name = k.replace("llm.model.", "")
            for kk in packed_modules_mapping:
                if kk in weight_name:
                    vv, shard_id = packed_modules_mapping[kk]
                    param_name = weight_name.replace(kk, vv)
                    try:
                        param = model.get_parameter(param_name)
                        weight_loader = param.weight_loader
                        weight_loader(param, v, shard_id)
                        break
                    except Exception as e:
                        print(e)
                        print(f"skip parameter (1): {weight_name}")
                        continue
            else:
                try:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, v)
                except Exception as e:
                    print(e)
                    print(f"skip parameter (2): {weight_name}")
                    continue
        else:
            missed += 1
            missed_names.append(weight_name)
            continue
    print(f"missed {missed} parameters: {missed_names}")

    # NOTE(xcsong): 2. merge text embedding, sos/taskid embedding, and speech embedding
    text_embedding_weight = embedding_weights["model.embed_tokens.weight"].cpu()  # [151936, 896]
    sos_taskid_embedding_weight = embedding_weights["llm_embedding.weight"].cpu()  # [2, 896]
    speech_embedding_weight = embedding_weights["speech_embedding.weight"].cpu()  # [6562, 896]
    final_embedding_weight = torch.cat([speech_embedding_weight, sos_taskid_embedding_weight, text_embedding_weight], dim=0)  # [158500, 896]
    param = model.get_parameter("model.embed_tokens.weight")
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, final_embedding_weight)


def load_model(model: nn.Module, path: str, hf_config: FunASRNanoConfig | None = None):
    load_llm_model_weights(model, path)
    # load_text_llm(model, path)
    # if model.model_type == "speech_llm":
    #     load_speech_llm(model, path, hf_config)
    # elif model.model_type == "text_llm":
    #     load_text_llm(model, path)
    # else:
    #     raise ValueError(f"Unsupported model type: {model.model_type}")

def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()}
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[to_device(v, device, dtype, non_blocking, copy) for v in dataclasses.astuple(data)]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[to_device(o, device, dtype, non_blocking, copy) for o in data])
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data