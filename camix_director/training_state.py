"""Safe full-state checkpoints and lightweight local telemetry."""
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .runtime import atomic_replace


class TrainingCancelled(Exception):pass


def save_torch(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(payload,temporary);atomic_replace(temporary,path)


def capture_rng(device):
    np_state=np.random.get_state()
    value=dict(python=random.getstate(),numpy=[np_state[0],np_state[1].tolist(),int(np_state[2]),int(np_state[3]),float(np_state[4])],torch=torch.get_rng_state())
    if device=='cuda':value['cuda']=torch.cuda.get_rng_state_all()
    if device=='mps' and hasattr(torch.mps,'get_rng_state'):value['mps']=torch.mps.get_rng_state()
    return value


def restore_rng(value,device):
    random.setstate(value['python']);n=value['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(value['torch'])
    if device=='cuda' and 'cuda' in value:torch.cuda.set_rng_state_all(value['cuda'])
    if device=='mps' and 'mps' in value:torch.mps.set_rng_state(value['mps'])


class Telemetry:
    def __init__(self,device):self.device=device;self.started=time.monotonic();self.cpu=time.process_time();self.mps_peak=0
    def sample(self):
        now=time.monotonic();cpu=time.process_time();elapsed=max(now-self.started,.001)
        result=dict(worker_cpu_percent=(cpu-self.cpu)/elapsed*100,host_dispatch_interval_seconds=elapsed)
        self.started=now;self.cpu=cpu
        try:
            import resource
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['rss_peak_bytes']=int(rss if __import__('sys').platform=='darwin' else rss*1024)
        except ImportError:result['rss_peak_bytes']=None
        if self.device=='cuda':result.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_reserved_bytes=torch.cuda.memory_reserved(),gpu_peak_bytes=torch.cuda.max_memory_allocated())
        if self.device=='mps':
            used=int(torch.mps.current_allocated_memory());self.mps_peak=max(self.mps_peak,used)
            result.update(gpu_allocated_bytes=used,gpu_peak_observed_bytes=self.mps_peak)
        return result
