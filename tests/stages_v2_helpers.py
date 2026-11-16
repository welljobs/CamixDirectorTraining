from pathlib import Path
import wave
import numpy as np
from camix_director.annotations_v2 import EventStore, import_v1_media
from camix_director.contracts_v2 import media_identity
from camix_director.corpus_v2 import CORPUS_FORMAT
from camix_director.features_v2 import extract
from camix_director.model_v2 import PolicyConfig
from camix_director.prepare_v2 import prepare_index
from camix_director.raw_demo import create
from v2_helpers import encoder_weights


def stage_fixture(root):
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    raw = root/'data/stage_fixture'
    paths = create(raw, seconds=6)
    weights, sha = encoder_weights(root)
    store = EventStore(root);entries=[];projects=[]
    for ordinal, path in enumerate(paths):
        # Generated negative-control data, not real professional directing.
        # Distinct PCM bytes across splits prevent exact-content leakage.
        for audio in Path(path).parent.glob('*.wav'):
            with wave.open(str(audio), 'rb') as f:
                params = f.getparams();samples = np.frombuffer(f.readframes(f.getnframes()), '<i2').copy()
            samples = (samples.astype(np.float64)*(1-ordinal*.13)).astype('<i2')
            with wave.open(str(audio), 'wb') as f:
                f.setparams(params);f.writeframes(samples.tobytes())
        m = import_v1_media(root, path)
        m['license'] = dict(name='generated synthetic test fixture', usage_confirmed=True)
        m['program_events'] = [dict(time_s=2.,camera_id='cam_b'),dict(time_s=4.,camera_id='cam_a')]
        m['shot_preferences'] = [dict(id=f's{i}',start_s=start,end_s=end,acceptable_cameras=[cam],cut_window_s=None,
                                      context_program=None,intent='subject',reason_tags=[]) for i,(start,end,cam) in enumerate([(0.,2.,'cam_a'),(2.,4.,'cam_b'),(4.,6.,'cam_a')])]
        m['events'] = [dict(id='marker',start_s=2.,end_s=4.,label='synthetic marker only',visible_cameras=['cam_b'],importance=.7)]
        p = store.create(m);m = p['manifest']
        m['comparisons'] = [dict(id='c1',start_s=1.,end_s=5.,source_identity=media_identity(m),
                                 a=dict(initial_program='cam_a',cuts=[]),
                                 b=dict(initial_program='cam_a',cuts=[dict(time_s=2.,camera_id='cam_b'),dict(time_s=4.,camera_id='cam_a')]),
                                 label='b',reviewer='automated synthetic fixture NOT human director',reason_tags=['coverage'],notes='fabricated test preference')]
        p = store.save(p['id'],p['revision'],m)
        p = store.review(p['id'],p['revision'],'automated synthetic fixture NOT human director',True,True)
        out = root/'runs/indexes'/p['id']/'index.json'
        index = prepare_index(root,p,out,decision_hz=2,chunk_steps=3,purpose='training',allow_synthetic=True)
        cache = extract(root,index,'v1_visual',str(weights),sha)
        entries.append(dict(index=out.relative_to(root).as_posix(),cache_id=cache['cache_id']));projects.append(p)
    config = PolicyConfig(visual_dim=16,max_cameras=3,hidden_dim=16,heads=4,layers=1,
                          short_tokens=4,shot_tokens=3,slow_tokens=2,slow_stride_s=1.,dropout=0.)
    return root,dict(format=CORPUS_FORMAT,name='synthetic-stage-fixture',entries=entries),config,projects
