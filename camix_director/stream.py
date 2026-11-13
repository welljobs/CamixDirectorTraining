"""Sequential inference boundary for a future stream adapter. Does not control Camix."""
import threading,math
import torch
from .config import INPUT_NAMES
from .train import load_checkpoint

class StreamingPolicy:
    def __init__(self,checkpoint):
        self.model,self.checkpoint=load_checkpoint(checkpoint)
        self.lock=threading.Lock();self.reset()

    def _reset_unlocked(self):
        self.memory,self.memory_mask=self.model.initial_memory()
        self.last_timestamp=None;self.generation=None

    def reset(self):
        with self.lock:self._reset_unlocked()

    @torch.no_grad()
    def step(self,features,timestamp,generation):
        with self.lock:
            if not math.isfinite(timestamp):raise ValueError("Non-finite timestamp")
            if generation!=self.generation:
                self._reset_unlocked();self.generation=generation
            if self.last_timestamp is not None and timestamp<=self.last_timestamp:
                raise ValueError('Duplicate/backward stream timestamp; reset on seek/reconnect')
            inputs=[features[name] for name in INPUT_NAMES[:-2]]+[self.memory,self.memory_mask]
            keep,take,gain,new_memory,new_mask=self.model(*inputs)
            self.memory,self.memory_mask=new_memory,new_mask;self.last_timestamp=timestamp
            return dict(keep_value=keep,take_value=take,switch_gain=gain,
                        qualification=self.checkpoint['qualification'],execute_allowed=False)
