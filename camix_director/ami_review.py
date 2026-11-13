"""Create unreviewed AMI review projects with source-resolution browser clips.
No training/labels/review approval is performed here; original AVI is retained.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from .ami_originals import MEETINGS,validate_entry,disk_guard,GIB
from .annotations import AnnotationStore
from .media_proxy import proxy_signature
from .runtime import atomic_json
from .storage import metadata_only


def source_resolution_clip(source,output,start,duration,fps=25):
    import av
    output=Path(output)
    if output.exists():raise ValueError('Do not overwrite existing review clip')
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.partial.mp4')
    count=0;last=-1
    with av.open(str(source)) as original,av.open(str(temporary),'w') as target:
        stream=original.streams.video[0]
        original.seek(int(start/float(stream.time_base)),stream=stream,backward=True)
        encoded=target.add_stream('libx264',rate=fps);encoded.width=stream.width;encoded.height=stream.height
        encoded.pix_fmt='yuv420p';encoded.options={'preset':'veryfast','crf':'18'}
        for frame in original.decode(stream):
            if frame.time is None or frame.time<start:continue
            if frame.time>=start+duration:break
            position=round((frame.time-start)*fps)
            if position<=last:continue
            last=position
            view=frame.reformat(width=stream.width,height=stream.height,format='yuv420p')
            view.pts=position;view.time_base=Fraction(1,fps)
            for packet in encoded.encode(view):target.mux(packet)
            count+=1
        for packet in encoded.encode():target.mux(packet)
    if not count:raise ValueError('Selected review interval contains no decoded video')
    temporary.replace(output);return count


def create(workspace,root):
    from PIL import Image,ImageDraw
    import av
    workspace=Path(workspace).resolve();root=Path(root).resolve()
    if not root.is_relative_to(workspace/'data'):raise ValueError('Keep review sources under workspace/data')
    source=json.loads((root/'source-manifest.json').read_text());prepared=json.loads((root/'preparation-report.json').read_text())
    store=AnnotationStore(workspace);receipt=root/'review-projects.json'
    known=json.loads(receipt.read_text()) if receipt.exists() else {'projects':[]}
    ids={p['meeting']:p['project_id'] for p in known['projects']};results=[]
    sheet=Image.new('RGB',(1440,1200),'#151c22');draw=ImageDraw.Draw(sheet)
    for mi,(meeting,split) in enumerate(MEETINGS):
        meta=metadata_only(Path(prepared['prepared_directory'])/(meeting+'_00.npz'))
        entries={(e['agent'],e['kind']):e for e in source['entries'] if e['meeting']==meeting}
        if meeting in ids:project=store.get(ids[meeting])
        else:
            cameras=[]
            for agent in 'ABCD':
                cameras.append(dict(id=agent,video=(root/validate_entry(entries[(agent,'video')])).relative_to(workspace).as_posix(),audio=(root/validate_entry(entries[(agent,'audio')])).relative_to(workspace).as_posix(),video_offset=0,audio_offset=0))
            project=store.create(dict(session_id=meeting,split=split,source='real',start_seconds=meta['source_start_seconds'],duration_seconds=meta['duration_seconds'],initial_program='A',cameras=cameras,scenario_tags=['meeting','AMI_original_576'],labels=[],program_events=[]))
            ids[meeting]=project['id'];atomic_json(receipt,dict(projects=[dict(meeting=k,project_id=v) for k,v in ids.items()],human_reviewed=False))
        if project.get('review'):raise ValueError('Existing project was reviewed by user; do not change it')
        session=project['session']
        if session.get('labels') or session.get('program_events') or session['start_seconds']!=meta['source_start_seconds'] or session['duration_seconds']!=meta['duration_seconds']:
            raise ValueError('User changed the review project; preserve it and create previews manually')
        for ci,camera in enumerate(session['cameras']):
            disk_guard(root,64*1024**2,5*GIB)
            output=root/'review'/meeting/camera['id']/'video.mp4'
            clip_identity=dict(source_sha256=camera['video_sha256'],start_seconds=session['start_seconds'],duration_seconds=session['duration_seconds'],source_resolution=[720,576],fps=25)
            identity_path=output.with_suffix('.identity.json')
            if output.exists() and (not identity_path.exists() or json.loads(identity_path.read_text())!=clip_identity):
                raise ValueError('Existing browser clip timing/source identity changed; do not silently reuse')
            if not output.exists():
                print(f"BROWSER CLIP {meeting} {camera['id']} 720x576 / 25fps",flush=True)
                source_resolution_clip(workspace/camera['video'],output,session['start_seconds'],session['duration_seconds'])
                atomic_json(identity_path,clip_identity)
            camera.update(preview_video=output.relative_to(workspace).as_posix(),proxy_signature=proxy_signature(session,camera))
            if mi==0:
                with av.open(str(output)) as container:
                    frame=next(container.decode(video=0));image=Image.fromarray(frame.to_ndarray(format='rgb24'))
                    x=(ci%2)*720;y=(ci//2)*600;sheet.paste(image,(x,y+24));draw.text((x+12,y+5),f"{meeting} - camera {camera['id']} - original 720x576",fill='white')
        project=store.save(project['id'],project['revision'],session)
        results.append(dict(meeting=meeting,split=split,project_id=project['id'],revision=project['revision'],human_reviewed=False))
    sheet.save(root/'original-four-camera-preview.jpg',quality=93)
    report=dict(projects=results,preview_resolution=[720,576],preview_fps=25,weak_training_packs_are_separate=True,human_reviewed=False,production_approved=False)
    atomic_json(receipt,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(create(a.workspace,a.root),ensure_ascii=False,indent=2))
