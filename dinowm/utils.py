import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from typing import Callable, Dict
import psutil
import hydra

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Memory usage in MB

def get_available_ram():
    mem = psutil.virtual_memory()
    return mem.available / (1024 * 1024 * 1024)  # Available memory in MB

def dict_to_namespace(cfg_dict):
    args = argparse.Namespace()
    for key in cfg_dict:
        setattr(args, key, cfg_dict[key])
    return args

def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct

def slice_trajdict_with_t(data_dict, start_idx=0, end_idx=None, step=1):
    if end_idx is None:
        end_idx = max(arr.shape[1] for arr in data_dict.values())
    return {key: arr[:, start_idx:end_idx:step, ...] for key, arr in data_dict.items()}

def concat_trajdict(dcts):
    full_dct = {}
    for k in dcts[0].keys():
        if isinstance(dcts[0][k], np.ndarray):
            full_dct[k] = np.concatenate([dct[k] for dct in dcts], axis=1)
        elif isinstance(dcts[0][k], torch.Tensor):
            full_dct[k] = torch.cat([dct[k] for dct in dcts], dim=1)
        else:
            raise TypeError(f"Unsupported data type: {type(dcts[0][k])}")
    return full_dct

def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def sample_tensors(tensors, n, indices=None):
    if indices is None:
        b = tensors[0].shape[0]
        indices = torch.randperm(b)[:n]
    indices = torch.tensor(indices)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            tensors[i] = tensor[indices]
    return tensors


def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(cfg_dict[key])
    return cfg_dict

def reduce_dict(f: Callable, d: Dict):
    return {k: reduce_dict(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model