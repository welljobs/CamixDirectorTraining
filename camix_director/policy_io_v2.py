"""Versioned V2 weights only; never loads code or promotes production approval."""
from pathlib import Path
import torch
from .contracts_v2 import fingerprint
from .model_v2 import CHECKPOINT_FORMAT, MODEL_FAMILY, FeatureDirector, PolicyConfig
from .storage import digest_file


def load_policy(path, device='cpu'):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 512*1024**2:
        raise ValueError('Missing/excessive V2 policy checkpoint')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != CHECKPOINT_FORMAT or payload.get('model_family') != MODEL_FAMILY:
        raise ValueError('Not a V2 feature policy checkpoint; V1/reward weights are not interchangeable')
    if payload.get('production_approved') is not False or payload.get('execute_allowed') is not False:
        raise ValueError('A research checkpoint cannot grant programme execution')
    if fingerprint(payload['feature_space']) != payload.get('feature_space_id'):
        raise ValueError('Policy encoder/preprocessing identity changed')
    config = PolicyConfig(**payload['config'])
    if config.visual_dim != payload['feature_space']['feature_dim']:
        raise ValueError('Policy feature dimension mismatch')
    model = FeatureDirector(config)
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval(), payload


def checkpoint_hash(path):
    return digest_file(Path(path))
