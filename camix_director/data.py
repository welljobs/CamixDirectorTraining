"""Validated sequence packs. Missing annotation is NaN, never an implicit KEEP."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import ModelConfig, SCHEMA_VERSION


def validate_pack(data,meta,config,allow_weak=False,path='dataset',inference_only=False):
    if meta.get('schema_version')!=SCHEMA_VERSION:raise ValueError(f'{path}: unsupported schema')
    for key in ('human_reviewed','alignment_verified'):
        if key in meta and not isinstance(meta[key],bool):raise ValueError(f'{path}: {key} must be a JSON boolean, not text')
    if meta.get('source') not in ('real','synthetic'):raise ValueError(f'{path}: explicit provenance required')
    if meta.get('split') not in ('train','validation','test'):raise ValueError(f'{path}: invalid split')
    for key in ('max_cameras','image_size','sample_rate','audio_samples','decision_hz'):
        if meta.get('config',{}).get(key)!=getattr(config,key):raise ValueError(f'{path}: media config mismatch: {key}')
    if meta['source']=='real' and (not meta.get('alignment_verified') or (not inference_only and not meta.get('human_reviewed') and not (allow_weak and meta.get('supervision','').startswith('weak_')))):
        raise ValueError(f'{path}: real data requires reviewed labels and verified alignment')
    t=data['frames'].shape[0];n,h,a=config.max_cameras,config.image_size,config.audio_samples
    if t<1:raise ValueError(f'{path}: empty sequence')
    if t>int(config.decision_hz*120)+1:raise ValueError('Episode longer than120s; split into bounded episodes first')
    shapes={'frames':(t,n,h,h,3),'audio':(t,n,a),'modality_mask':(t,n,2),'camera_mask':(t,n),'media_age':(t,n,2),'current_program':(t,n),'held_seconds':(t,1),'preferences':(t,n+1),'times':(t,)}
    for key,shape in shapes.items():
        if key not in data or data[key].shape!=shape:raise ValueError(f'{path}: {key} expected {shape}')
        if key=='frames':
            if data[key].dtype!=np.uint8:raise ValueError('uint8 RGB required')
            continue
        # Bound validation temporaries independently of episode length.
        for start in range(0,t,16):
            chunk=data[key][start:start+16]
            if key=='preferences':
                if np.isinf(chunk).any():raise ValueError('Infinite preferences')
            elif not np.isfinite(chunk).all():raise ValueError(f'Non-finite {key}')
            if key=='audio' and np.abs(chunk).max()>1.001:raise ValueError('PCM must be in [-1,1]')
            if key in ('modality_mask','camera_mask','current_program') and not np.isin(chunk,[0,1]).all():raise ValueError(f'{key} must be binary')
            if key in ('media_age','held_seconds') and (chunk<0).any():raise ValueError(f'Negative {key}')
    if not np.all(np.diff(data['times'])>0):raise ValueError('Strictly increasing timestamps required')
    if (data['current_program'].sum(-1)>1).any():raise ValueError('Invalid programme one-hot')
    if (data['camera_mask']>data['modality_mask'][:,:,0]).any():raise ValueError('Selectable camera without valid video')
    return data,meta


def read_pack(path,config,allow_weak=False,cache_root=None,mapped=False,inference_only=False):
    from .storage import mapped_pack,digest_file
    if mapped:
        data,meta=mapped_pack(path,cache_root)
    else:
        with np.load(path,allow_pickle=False) as source:
            data={key:source[key] for key in source.files if key!='metadata'}
            meta=json.loads(str(source['metadata'].item()))
        meta.update(pack_file=Path(path).name,pack_sha256=digest_file(path))
    return validate_pack(data,meta,config,allow_weak,path,inference_only)


def has_comparison(preferences):
    finite = np.isfinite(preferences)
    safe = np.nan_to_num(preferences)
    return bool(((safe[:,None] > safe[None,:]) & finite[:,None] & finite[None,:]).any())


class SessionDataset(Dataset):
    def __init__(self, root, split, config, sequence_steps=16, burn_in=4, allow_synthetic=False, allow_weak=False, cache_root=None):
        self.config, self.burn_in, self.sequence_steps = config, burn_in, sequence_steps
        if sequence_steps < 2 or not 0 <= burn_in < sequence_steps:
            raise ValueError("burn-in must be shorter than sequence")
        self.packs, self.metadata, self.windows = [], [], []
        from .storage import dataset_paths,metadata_only,mapped_pack,MMapPool
        paths=dataset_paths(root);pool=MMapPool(20);all_sessions={};source_splits={}
        for path in paths:
            manifest_meta=metadata_only(path);session=manifest_meta.get('session_id');part=manifest_meta.get('split')
            if session in all_sessions and all_sessions[session]!=part:raise ValueError('Session leakage across train/validation/test')
            all_sessions[session]=part
            for item in (manifest_meta.get('source_media_sha256') or {}).values():
                identity=item.get('video')
                if identity:
                    if identity in source_splits and source_splits[identity]!=part:raise ValueError('Source-video leakage across train/validation/test; preserve the original recording ID')
                    source_splits[identity]=part
            if part!=split:continue
            data,meta=mapped_pack(path,cache_root,pool)
            if meta.get("preview_only"):raise ValueError("Preview-only data cannot train; annotate and review a supervised revision first")
            validate_pack(data,meta,config,allow_weak,path)
            if meta["source"] == "synthetic" and not allow_synthetic:
                raise ValueError("Synthetic data is refused unless --allow-synthetic is explicit")
            sid = meta.get("session_id")
            if not sid:
                raise ValueError("Original-event session_id is required")
            index = len(self.packs)
            self.packs.append(data); self.metadata.append(meta)
            length = data["times"].shape[0]
            if length < sequence_steps:
                raise ValueError(f"{path}: sequence is too short")
            starts = set(range(0,length-sequence_steps+1,max(1,sequence_steps//2)))
            starts.add(length-sequence_steps)
            for start in sorted(starts):
                if any(has_comparison(x) for x in data["preferences"][start+burn_in:start+sequence_steps]):
                    self.windows.append((index,start))
        if not self.windows:
            raise ValueError(f"No supervised {split} sequences in {root}; supply synchronized media and preferences")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        pack, start = self.windows[index]
        source = self.packs[pack]
        out = {}
        for key in ("frames","audio","modality_mask","camera_mask","media_age","current_program","held_seconds","preferences"):
            value = source[key][start:start+self.sequence_steps].copy()
            if key == "frames":
                value = value.transpose(0,1,4,2,3).astype(np.float32) / 127.5 - 1
                key = "video_frames"
            elif key == "audio":
                key = "audio_pcm"
            out[key] = torch.from_numpy(value.astype(np.float32,copy=False))
        return out

    def provenance(self):
        return [{key: m.get(key) for key in ("session_id","split","source","human_reviewed","alignment_verified","supervision","program_history_source","pack_file","pack_sha256","license","weak_label_policy","review","annotation_version","scenario_tags","source_media_sha256")} for m in self.metadata]


def validate_split_separation(*datasets):
    seen = set()
    for dataset in datasets:
        ids = {m["session_id"] for m in dataset.metadata}
        if ids & seen:
            raise ValueError("Session leakage across training/validation/test")
        seen |= ids
