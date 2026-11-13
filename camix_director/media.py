"""Timestamp-based causal frame and PCM windows using isolated PyAV libraries."""
import numpy as np
import io,urllib.request
from collections import OrderedDict
from contextlib import contextmanager

class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded-memory HTTP reader. Never downloads an entire AVI implicitly."""
    def __init__(self,url,block_size=512*1024,max_blocks=16):
        self.url=url;self.block_size=block_size;self.max_blocks=max_blocks;self.position=0
        self.cache=OrderedDict();self.network_bytes=0
        with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=30) as r:
            self.length=int(r.headers.get("Content-Length",0));self.etag=r.headers.get("ETag")
        if not 0<self.length<512*1024*1024:raise ValueError("Missing/excessive remote media size")
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.length+offset if whence==2 else -1
        if target<0:raise ValueError("negative/unsupported seek")
        self.position=target;return target
    def read(self,size=-1):
        if self.position>=self.length:return b""
        if size<0:raise ValueError("Unbounded remote reads are disabled")
        remaining=min(size,self.length-self.position);parts=[]
        while remaining:
            block=self.position//self.block_size;offset=self.position%self.block_size
            if block not in self.cache:
                start=block*self.block_size;end=min(self.length,start+self.block_size)-1
                headers={"Range":f"bytes={start}-{end}","User-Agent":"CamixDirectorTraining/0.1"}
                if self.etag:headers["If-Range"]=self.etag
                last_error=None
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(urllib.request.Request(self.url,headers=headers),timeout=45) as r:
                            if r.status!=206 or not r.headers.get("Content-Range","").startswith(f"bytes {start}-{end}/"):
                                raise ValueError("Remote server did not honour byte range")
                            data=r.read(end-start+2)
                        if len(data)!=end-start+1:raise IOError("Truncated media range")
                        self.network_bytes+=len(data);self.cache[block]=data;last_error=None;break
                    except Exception as error:last_error=error
                if last_error:raise last_error
                while len(self.cache)>self.max_blocks:self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            count=min(remaining,len(self.cache[block])-offset)
            parts.append(self.cache[block][offset:offset+count]);remaining-=count;self.position+=count
        return b"".join(parts)

@contextmanager
def open_media(source):
    import av
    reader=HTTPRangeReader(str(source)) if str(source).startswith(("http://","https://")) else None
    container=av.open(reader if reader is not None else str(source))
    try:yield container
    finally:
        container.close()
        if reader is not None:reader.close()


def video_samples(url,start,times,size):
    import av
    frames=np.zeros((len(times),size,size,3),np.uint8);ages=np.full(len(times),10,np.float32)
    with open_media(url) as container:
        stream=container.streams.video[0]
        # File-like range access gives AVI a normal seekable source.
        container.seek(int(max(0,start-1)/float(stream.time_base)),stream=stream,backward=True)
        decoder=iter(container.decode(stream));lookahead=next(decoder,None);previous=None
        for i,relative in enumerate(times):
            target=start+relative
            while lookahead is not None and lookahead.time is not None and lookahead.time<=target+1e-7:
                previous=lookahead;lookahead=next(decoder,None)
            if previous is None:continue
            ages[i]=max(0,target-previous.time)
            scale=min(size/previous.width,size/previous.height)
            width=max(1,round(previous.width*scale));height=max(1,round(previous.height*scale))
            image=previous.reformat(width=width,height=height,format='rgb24').to_ndarray()
            left=(size-width)//2;top=(size-height)//2
            frames[i,top:top+height,left:left+width]=image
    return frames,(ages<=1).astype(np.float32),ages


def audio_samples(url,start,times,config):
    import av
    base=start-config.audio_samples/config.sample_rate-1
    end_time=start+float(times[-1])+.1
    values=np.zeros(int(np.ceil((end_time-base)*config.sample_rate)),np.float32)
    coverage=np.zeros(len(values),bool)
    with open_media(url) as container:
        stream=container.streams.audio[0]
        container.seek(int(max(0,base)/float(stream.time_base)),stream=stream,backward=True)
        resampler=None
        for frame in container.decode(stream):
            if frame.time is None:raise ValueError("Audio has no timestamp")
            if frame.time>end_time:break
            if frame.sample_rate!=config.sample_rate or len(frame.layout.channels)!=1:
                if resampler is None:resampler=av.AudioResampler(format='fltp',layout='mono',rate=config.sample_rate)
                items=resampler.resample(frame)
            else:items=[frame]
            for out in items:
                if out.time is None:raise ValueError("Resampled audio has no timestamp")
                samples=out.to_ndarray().reshape(-1)
                if np.issubdtype(samples.dtype,np.integer):samples=samples.astype(np.float32)/float(np.iinfo(samples.dtype).max+1)
                samples=samples.astype(np.float32)
                index=round((out.time-base)*config.sample_rate)
                lo=max(0,index);hi=min(len(values),index+len(samples))
                if hi>lo:values[lo:hi]=samples[lo-index:hi-index];coverage[lo:hi]=True
    audio=np.zeros((len(times),config.audio_samples),np.float32);valid=np.zeros(len(times),np.float32)
    for i,t in enumerate(times):
        end=round((start+t-base)*config.sample_rate);begin=end-config.audio_samples
        if begin>=0 and end<=len(values) and coverage[begin:end].all():
            audio[i]=np.clip(values[begin:end],-1,1);valid[i]=1
    return audio,valid


