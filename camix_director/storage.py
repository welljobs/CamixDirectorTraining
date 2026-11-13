"""Content-addressed, disk-backed sequence cache. Never unpickles dataset input."""
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import zipfile
import numpy as np
from .runtime import atomic_json, file_lock

FIELDS=('frames','audio','modality_mask','camera_mask','media_age','current_program','held_seconds','preferences','times')


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def metadata_only(path):
    with zipfile.ZipFile(path) as archive:
        item=archive.getinfo('metadata.npy')
        if item.file_size>1024*1024:raise ValueError('Dataset metadata exceeds1MiB')
        meta=json.loads(str(np.load(io.BytesIO(archive.read(item)),allow_pickle=False).item()))
    if not isinstance(meta,dict):raise ValueError('Invalid dataset metadata')
    return meta


def dataset_paths(root,verify=True):
    root=Path(root).resolve();manifest=root/'dataset-manifest.json'
    if not manifest.exists():return sorted(root.glob('*.npz'))
    content=json.loads(manifest.read_text(encoding='utf-8'));paths=[]
    if content.get('format')!='camix_dataset_version_v1':raise ValueError('Unknown dataset version')
    from .lineage import canonical_hash
    expected='v-'+canonical_hash(dict(purpose=content['purpose'],files=sorted(item['sha256'] for item in content['files'])))[:20]
    if content.get('id')!=expected or root.name!=expected:raise ValueError('Frozen dataset manifest identity changed')
    for item in content['files']:
        path=(root/item['object_path']).resolve()
        if not path.is_relative_to(root.parent.parent/'objects'):raise ValueError('Dataset object escapes object store')
        if verify and digest_file(path)!=item['sha256']:raise ValueError('Frozen dataset object changed: '+str(path))
        paths.append(path)
    return paths


class MMapPool:
    def __init__(self,capacity=20):self.capacity=capacity;self.maps=OrderedDict()
    def get(self,path):
        key=str(path)
        if key not in self.maps:self.maps[key]=np.load(path,allow_pickle=False,mmap_mode='r')
        self.maps.move_to_end(key)
        while len(self.maps)>self.capacity:self.maps.popitem(last=False)
        return self.maps[key]
    def __getstate__(self):return {'capacity':self.capacity,'maps':OrderedDict()}


class MappedPack(Mapping):
    def __init__(self,directory,pool=None):self.directory=Path(directory);self.pool=pool or MMapPool()
    def __getitem__(self,key):
        if key not in FIELDS:raise KeyError(key)
        return self.pool.get(self.directory/(key+'.npy'))
    def __iter__(self):return iter(FIELDS)
    def __len__(self):return len(FIELDS)


def mapped_pack(path,cache_root=None,pool=None):
    path=Path(path).resolve();sha=digest_file(path)
    cache_root=Path(cache_root or path.parent/'.camix-cache')
    directory=cache_root/sha;marker=directory/'cache.json'
    with file_lock(cache_root/(sha+'.lock')):
        if not marker.exists():
            directory.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                infos={name:archive.getinfo(name+'.npy') for name in FIELDS}
                needed=sum(item.file_size for item in infos.values())
                remaining=sum(item.file_size for name,item in infos.items() if not (directory/(name+'.npy')).exists())
                if remaining+256*1024*1024>shutil.disk_usage(directory).free:raise ValueError('Insufficient disk for bounded-memory cache; free space or choose another cache directory')
                if needed>512*1024**3:raise ValueError('One episode expands beyond512GiB; shard it into shorter episodes')
                for name,item in infos.items():
                    final=directory/(name+'.npy');temporary=directory/(name+'.npy.partial')
                    # Re-extract incomplete prior attempts; never trust an uncommitted member.
                    if final.exists():final.unlink()
                    with archive.open(item) as source,temporary.open('wb') as target:shutil.copyfileobj(source,target,1024*1024)
                    array=np.load(temporary,allow_pickle=False,mmap_mode='r')
                    if array.dtype.hasobject:raise ValueError('Object arrays are forbidden')
                    del array;temporary.replace(final)
                atomic_json(marker,dict(format='camix_mmap_cache_v1',source_sha256=sha,raw_bytes=needed))
    metadata=metadata_only(path);metadata['pack_file']=path.name;metadata['pack_sha256']=sha
    return MappedPack(directory,pool),metadata
