"""Bounded local media inspection and waveform summaries for annotation."""
from pathlib import Path
import math
import numpy as np


def probe(path):
    import av
    with av.open(str(path)) as container:
        streams=[]
        for stream in container.streams:
            duration=float(stream.duration*stream.time_base) if stream.duration is not None and stream.time_base else None
            item=dict(type=stream.type,codec=stream.codec_context.name,duration_seconds=duration)
            if stream.type=='video':item.update(width=stream.width,height=stream.height,fps=float(stream.average_rate) if stream.average_rate else None)
            if stream.type=='audio':item.update(sample_rate=stream.codec_context.sample_rate,channels=stream.codec_context.channels)
            streams.append(item)
        total=float(container.duration/av.time_base) if container.duration else max([s['duration_seconds'] or 0 for s in streams],default=0)
    return dict(path=str(path),duration_seconds=total,streams=streams,browser_hint='MP4/H264 and WAV are recommended; unsupported codecs need a proxy, not a renamed extension')


def waveform(path,start=0,duration=60,bins=240):
    import av
    if not 0<duration<=120 or not math.isfinite(start):raise ValueError('Waveform window must be at most120s')
    sums=np.zeros(bins,np.float64);counts=np.zeros(bins,np.int64)
    with av.open(str(path)) as container:
        if not container.streams.audio:raise ValueError('No audio stream')
        stream=container.streams.audio[0];container.seek(int(max(0,start)/float(stream.time_base)),stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.time is None:continue
            if frame.time>start+duration:break
            values=frame.to_ndarray().astype(np.float64)
            if frame.format.is_planar:values=values.mean(0)
            else:values=values.reshape(-1,len(frame.layout.channels)).mean(1)
            if frame.format.name.startswith(('s16','s32','u8')):values/=float(32768 if '16' in frame.format.name else 2147483648 if '32' in frame.format.name else 128)
            times=frame.time+np.arange(len(values))/frame.sample_rate
            indices=((times-start)/duration*bins).astype(np.int64);valid=(indices>=0)&(indices<bins)&(times>=start)
            np.add.at(sums,indices[valid],values[valid]**2);np.add.at(counts,indices[valid],1)
    rms=np.sqrt(sums/np.maximum(counts,1));maximum=max(float(rms.max()),1e-10)
    return dict(start_seconds=start,duration_seconds=duration,rms=rms.tolist(),normalized=(rms/maximum).tolist(),warning='Waveform is a sync aid, not automatic proof of clock alignment')
