"""Synthetic raw-media fixtures for learning the annotation UI, never real director data."""
import argparse,json
from pathlib import Path
import wave
import numpy as np
from PIL import Image,ImageDraw
from .runtime import atomic_json


def create(root,seconds=12):
    import av
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    projects=[]
    for index,split in enumerate(('train','validation','test')):
        directory=root/split;directory.mkdir();cameras=[]
        for slot,identifier in enumerate(('cam_a','cam_b')):
            filename=directory/(identifier+'.mp4')
            with av.open(str(filename),'w') as container:
                stream=container.add_stream('libx264',rate=10);stream.width=320;stream.height=180;stream.pix_fmt='yuv420p'
                for frame_index in range(seconds*10):
                    image=Image.new('RGB',(320,180),(20+slot*20,45,60+index*20));draw=ImageDraw.Draw(image)
                    x=(frame_index*3+slot*70)%270;draw.ellipse((x,65,x+45,110),fill=(220,145,65) if slot==0 else (75,195,160))
                    draw.text((12,12),'SYNTHETIC DEMO - NOT DIRECTOR GROUND TRUTH',fill='white');draw.text((12,145),f'{split}  {identifier}  {frame_index/10:.1f}s',fill='white')
                    frame=av.VideoFrame.from_ndarray(np.array(image),format='rgb24')
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            rate=16000;t=np.arange(seconds*rate)/rate;values=(np.sin(2*np.pi*(220+slot*110)*t)*(.05+.25*((t.astype(int)//2+slot)%2))).astype(np.float32)
            audio=directory/(identifier+'.wav')
            with wave.open(str(audio),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes((values*32767).astype('<i2').tobytes())
            cameras.append(dict(id=identifier,video=filename.name,audio=audio.name,video_offset=0,audio_offset=0))
        (directory/'preferences.jsonl').write_text('',encoding='utf-8');(directory/'program-events.jsonl').write_text('',encoding='utf-8')
        manifest=dict(session_id='SYNTHETIC-RAW-'+split,split=split,source='synthetic',duration_seconds=seconds,start_seconds=0,initial_program='cam_a',cameras=cameras,alignment_verified=False,human_reviewed=False,scenario_tags=['synthetic_demo_only'],annotations='preferences.jsonl',program_events='program-events.jsonl',provenance=dict(origin='generated synthetic UI demonstration; not real media'))
        atomic_json(directory/'session.json',manifest);projects.append(str(directory/'session.json'))
    atomic_json(root/'raw-demo.json',dict(source='synthetic',manifests=projects,warning='Only validates annotation/prepare/train workflow, not directing quality'))
    return projects

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seconds',type=int,default=12);p.add_argument('--accept-synthetic',action='store_true');a=p.parse_args()
    if not a.accept_synthetic:p.error('Explicit synthetic acknowledgement required')
    if not 6<=a.seconds<=120:p.error('Demo seconds6..120')
    print(json.dumps(create(a.root,a.seconds),indent=2))
