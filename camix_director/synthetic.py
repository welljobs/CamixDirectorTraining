"""Deliberately artificial audiovisual fixtures. NEVER production supervision."""
import json
from pathlib import Path
import numpy as np
from .config import ModelConfig, SCHEMA_VERSION


def create_fixture(root, config, steps=40, sessions=3, seed=19):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    if steps < 4 or sessions < 3:
        raise ValueError("Need at least 3 separate synthetic sessions")
    for split_index, split in enumerate(("train","validation","test")):
        for session in range(sessions if split == "train" else 1):
            rng = np.random.default_rng(seed + split_index*1000 + session)
            n,h,s = config.max_cameras,config.image_size,config.audio_samples
            frames = rng.integers(0,20,(steps,n,h,h,3),dtype=np.uint8)
            audio = np.zeros((steps,n,s),np.float32)
            valid = np.ones((steps,n),np.float32)
            modality = np.ones((steps,n,2),np.float32)
            program = np.zeros((steps,n),np.float32);program[:,0]=1
            prefs = np.full((steps,n+1),np.nan,np.float32)
            x = np.arange(s,dtype=np.float32)/config.sample_rate
            for t in range(steps):
                winner = (t//8 + session) % n
                for camera in range(n):
                    frames[t,camera] = np.clip(frames[t,camera].astype(np.int16) + (185 if camera==winner else 35),0,255)
                    audio[t,camera] = (0.4 if camera==winner else 0.03)*np.sin(2*np.pi*(220+camera*60)*x)
                prefs[t,0] = 1.0 if winner==0 else 0.0
                prefs[t,1:] = 0.0
                prefs[t,winner+1] = 1.0
            meta = dict(schema_version=SCHEMA_VERSION,session_id=f"SYNTHETIC-{split}-{session}",split=split,
                        source="synthetic",human_reviewed=False,alignment_verified=False,
                        warning="Artificial color/tone task only; not director quality",config=config.to_dict())
            np.savez_compressed(root/f"{split}_{session}.npz",frames=frames,audio=audio,
                modality_mask=modality,camera_mask=valid,media_age=np.zeros((steps,n,2),np.float32),
                current_program=program,held_seconds=np.arange(steps,dtype=np.float32)[:,None]/config.decision_hz,
                preferences=prefs,times=np.arange(steps,dtype=np.float64)/config.decision_hz,
                metadata=np.array(json.dumps(meta)))
    return root

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description='Artificial fixtures ONLY; never real-world director training')
    p.add_argument('--output',required=True);p.add_argument('--config',required=True)
    p.add_argument('--steps',type=int,default=40);p.add_argument('--sessions',type=int,default=3)
    p.add_argument('--acknowledge-synthetic',action='store_true')
    args=p.parse_args()
    if not args.acknowledge_synthetic:p.error('Pass --acknowledge-synthetic; fixtures are not real director supervision')
    create_fixture(args.output,ModelConfig(**json.loads(Path(args.config).read_text(encoding="utf-8"))),args.steps,args.sessions)
