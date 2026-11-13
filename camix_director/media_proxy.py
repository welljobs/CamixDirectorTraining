"""Browser proxy for a bounded annotation episode; original media remains training source."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import wave
import numpy as np
from .annotations import AnnotationStore
from .lineage import canonical_hash
from .runtime import atomic_json
from .storage import digest_file


def proxy_signature(session,camera):
    return canonical_hash(dict(video_sha256=camera.get('video_sha256'),audio_sha256=camera.get('audio_sha256'),start=session.get('start_seconds',0),duration=session['duration_seconds'],video_offset=camera.get('video_offset',0),audio_offset=camera.get('audio_offset',0)))


def make_proxy(workspace,project_id,camera_index,output,stop_file=None):
    import av
    root=Path(workspace).resolve();project=AnnotationStore(root).get(project_id);session=project['session'];camera=session['cameras'][camera_index]
    output=Path(output);output.mkdir(parents=True,exist_ok=False);duration=float(session['duration_seconds']);start=session['start_seconds']+camera.get('video_offset',0)
    if start<0:raise ValueError('Proxy source start is negative; choose a valid segment/offset first')
    video_path=root/camera['video']
    if digest_file(video_path)!=camera['video_sha256']:raise ValueError('Original media changed; reload/review source identity first')
    count=0;fps=15
    with av.open(str(video_path)) as source,av.open(str(output/'video.mp4'),'w') as target:
        stream=source.streams.video[0];source.seek(int(start/float(stream.time_base)),stream=stream,backward=True)
        scale=min(960/stream.width,540/stream.height,1);width=max(2,int(stream.width*scale)//2*2);height=max(2,int(stream.height*scale)//2*2)
        encoded=target.add_stream('libx264',rate=fps);encoded.width=width;encoded.height=height;encoded.pix_fmt='yuv420p';encoded.options={'preset':'veryfast','crf':'22'};last=-1
        for frame in source.decode(stream):
            if stop_file and Path(stop_file).exists():raise InterruptedError('Proxy cancelled')
            if frame.time is None or frame.time<start:continue
            if frame.time>=start+duration:break
            position=round((frame.time-start)*fps)
            if position<=last:continue
            last=position;render=frame.reformat(width=width,height=height,format='yuv420p');render.pts=position;render.time_base=Fraction(1,fps)
            for packet in encoded.encode(render):target.mux(packet)
            count+=1
        for packet in encoded.encode():target.mux(packet)
    if not count:raise ValueError('No video frames in selected proxy interval')
    audio_file=None;coverage_ratio=None
    if camera.get('audio'):
        audio_path=root/camera['audio'];audio_start=session['start_seconds']+camera.get('audio_offset',0);rate=16000
        if digest_file(audio_path)!=camera['audio_sha256']:raise ValueError('Audio source changed')
        pcm=np.zeros(round(duration*rate),np.float32);coverage=np.zeros(len(pcm),bool)
        with av.open(str(audio_path)) as container:
            stream=container.streams.audio[0];container.seek(int(max(0,audio_start)/float(stream.time_base)),stream=stream,backward=True);resampler=av.AudioResampler(format='fltp',layout='mono',rate=rate)
            for frame in container.decode(stream):
                if stop_file and Path(stop_file).exists():raise InterruptedError('Proxy cancelled')
                if frame.time is not None and frame.time>audio_start+duration+.2:break
                for block in resampler.resample(frame):
                    if block.time is None:continue
                    values=block.to_ndarray().reshape(-1);begin=round((block.time-audio_start)*rate);lo=max(0,begin);hi=min(len(pcm),begin+len(values))
                    if hi>lo:pcm[lo:hi]=values[lo-begin:hi-begin];coverage[lo:hi]=True
        audio_file='audio.wav';coverage_ratio=float(coverage.mean())
        with wave.open(str(output/audio_file),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
    report=dict(format='camix_browser_proxy_v1',project_id=project_id,camera_index=camera_index,source_signature=proxy_signature(session,camera),preview_video=(output/'video.mp4').relative_to(root).as_posix(),preview_audio=(output/audio_file).relative_to(root).as_posix() if audio_file else None,video_frames=count,audio_coverage=coverage_ratio,warning='Proxy is compressed review media only. Source clock hashes/offsets bind it; training still uses original media.')
    atomic_json(output/'proxy-report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--project',required=True);p.add_argument('--camera',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--stop-file');a=p.parse_args();print(json.dumps(make_proxy(a.workspace,a.project,a.camera,a.output,a.stop_file),indent=2))
