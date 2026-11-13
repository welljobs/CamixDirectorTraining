"""Offline QA: camera contact sheet + mask/PCM/label statistics. No downloads."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .config import ModelConfig
from .data import read_pack


def inspect(data_root,config,out,allow_weak=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);report=[]
    for path in sorted(Path(data_root).glob('*.npz')):
        data,meta=read_pack(path,config,allow_weak)
        indices=np.linspace(0,len(data['times'])-1,6,dtype=int)
        sheet=Image.new('RGB',(6*180,config.max_cameras*180+50),'#191d24');draw=ImageDraw.Draw(sheet)
        draw.text((10,8),f"{meta['session_id']} | {meta['split']} | source={meta['source']}",fill='white')
        for col,index in enumerate(indices):
            for camera in range(config.max_cameras):
                image=Image.fromarray(data['frames'][index,camera]).resize((160,160))
                x=col*180+10;y=camera*180+48;sheet.paste(image,(x,y))
                label=data['preferences'][index].tolist()
                draw.text((x,y-14),f"t={data['times'][index]:.1f}s cam{camera}",fill='white')
        image_name=f"{path.stem}-contact-sheet.jpg";sheet.save(out/image_name,quality=90)
        rms=np.sqrt(np.mean(data['audio']**2,axis=-1))
        prefs=data['preferences'];keep=prefs[:,0];selected=prefs[:,1:]
        report.append(dict(session_id=meta['session_id'],split=meta['split'],frames=len(prefs),
            camera_ids=meta.get('camera_ids',[]),labelled_ticks=int(np.isfinite(keep).sum()),
            prefer_keep_ticks=int((keep==1).sum()),prefer_other_ticks=int((keep==0).sum()),
            video_valid_ratio=data['modality_mask'][:,:,0].mean(0).tolist(),
            audio_valid_ratio=data['modality_mask'][:,:,1].mean(0).tolist(),
            audio_rms_mean=rms.mean(0).tolist(),max_video_age=float(data['media_age'][:,:,0].max()),
            rgb_min=int(data['frames'].min()),rgb_max=int(data['frames'].max()),contact_sheet=image_name,
            warning='Descriptive preprocessing checks, not human acceptance of source alignment or director quality.'))
    (out/'data-qa.json').write_text(json.dumps(report,indent=2), encoding="utf-8")
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',required=True)
    p.add_argument('--out',required=True);p.add_argument('--allow-weak-supervision',action='store_true');a=p.parse_args()
    print(json.dumps(inspect(a.data,ModelConfig(**json.loads(Path(a.config).read_text(encoding="utf-8"))),a.out,a.allow_weak_supervision),indent=2))
