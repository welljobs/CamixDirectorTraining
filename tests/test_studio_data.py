import json
from pathlib import Path
import numpy as np
import pytest
from camix_director.config import ModelConfig
from camix_director.synthetic import create_fixture
from camix_director.dataset_tools import quality,freeze
from camix_director.storage import dataset_paths,digest_file
from camix_director.annotations import AnnotationStore
from camix_director.prepare import prepare

@pytest.fixture
def cfg():return ModelConfig(max_cameras=2,image_size=16,audio_samples=256,hidden_dim=16,layers=1,memory_tokens=4,dropout=0)


def media_fixture(root):
    import av,wave
    base=root/'data/raw';base.mkdir(parents=True)
    for name in ('a','b'):
        with av.open(str(base/(name+'.mp4')),'w') as container:
            stream=container.add_stream('libx264',rate=5);stream.width=32;stream.height=32;stream.pix_fmt='yuv420p'
            for i in range(40):
                frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i*4,np.uint8),format='rgb24')
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        with wave.open(str(base/(name+'.wav')),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((np.sin(np.arange(128000)*.04)*10000).astype('<i2').tobytes())
    return dict(session_id='synthetic-raw-event',split='train',source='synthetic',start_seconds=0,duration_seconds=6,initial_program='a',scenario_tags=['test-only'],cameras=[dict(id=n,video=f'data/raw/{n}.mp4',audio=f'data/raw/{n}.wav') for n in ('a','b')],labels=[],program_events=[])


def test_freeze_is_content_addressed_and_checks_corruption(tmp_path,cfg):
    data=tmp_path/'data/prepared';create_fixture(data,cfg,steps=12,sessions=3)
    result=freeze(data,tmp_path,'v1',allow_synthetic=True)
    version=tmp_path/'datasets/versions'/result['id']
    paths=dataset_paths(version);assert len(paths)==5
    again=freeze(data,tmp_path,'same data',allow_synthetic=True);assert again['id']==result['id']
    target=paths[0];target.chmod(0o644);target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):dataset_paths(version)


def test_quality_detects_renamed_content_leakage(tmp_path,cfg):
    create_fixture(tmp_path,cfg,steps=12,sessions=3)
    with np.load(tmp_path/'train_0.npz',allow_pickle=False) as z:payload={k:z[k] for k in z.files}
    metadata=json.loads(str(payload['metadata'].item()));metadata.update(session_id='renamed-same-footage',split='test');payload['metadata']=np.array(json.dumps(metadata));np.savez_compressed(tmp_path/'leak.npz',**payload)
    report=quality(tmp_path);assert not report['passed'] and any('相同音视频' in e for e in report['errors'])


def test_annotation_review_revision_and_prepare_integrity(tmp_path,cfg):
    store=AnnotationStore(tmp_path);session=media_fixture(tmp_path);project=store.create(session)
    session=project['session'];session['labels']=[dict(id='l1',time_s=2,preferred='b',rejected=['keep'])]
    draft=store.save(project['id'],project['revision'],session)
    with pytest.raises(ValueError,match='另一页面'):store.save(project['id'],1,session)
    with pytest.raises(ValueError,match='分别确认'):store.review(draft['id'],draft['revision'],'tester',False,True)
    reviewed=store.review(draft['id'],draft['revision'],'GUI synthetic tester',True,True)
    manifest=tmp_path/reviewed['export_manifest']
    assert reviewed['session']['cameras'][0]['video']=='data/raw/a.mp4'
    result=prepare(manifest,tmp_path/'data/prepared/a.npz',cfg)
    with np.load(tmp_path/'data/prepared/a.npz',allow_pickle=False) as z:m=json.loads(str(z['metadata'].item()))
    assert m['source']=='synthetic' and m['human_reviewed'] and m['review']['content_sha256']
    labels=manifest.parent/'preferences.jsonl';rows=[json.loads(s) for s in labels.read_text().splitlines()];rows[0]['preferred']='a';labels.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='changed after review'):prepare(manifest,tmp_path/'data/prepared/tampered.npz',cfg)
    edited=store.save(reviewed['id'],reviewed['revision'],reviewed['session']);assert edited['review'] is None


def test_source_provenance_cannot_be_changed_in_editor(tmp_path):
    store=AnnotationStore(tmp_path);p=store.create(media_fixture(tmp_path));session=p['session'];session['source']='real'
    with pytest.raises(ValueError,match='来源类型'):store.save(p['id'],p['revision'],session)

def test_frozen_manifest_identity_is_verified(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    result=freeze(data,tmp_path,allow_synthetic=True);directory=tmp_path/'datasets/versions'/result['id'];p=directory/'dataset-manifest.json'
    value=json.loads(p.read_text());value['purpose']='evaluation';p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='manifest identity'):dataset_paths(directory)

def test_unlabelled_preview_can_replay_but_cannot_become_training_data(tmp_path,cfg):
    from camix_director.data import read_pack,SessionDataset
    store=AnnotationStore(tmp_path);project=store.create(media_fixture(tmp_path))
    preview=store.export_preview(project['id'],project['revision'],True)
    out=tmp_path/'data/preview/session.npz';prepare(tmp_path/preview['manifest'],out,cfg)
    data,meta=read_pack(out,cfg,inference_only=True)
    assert meta['preview_only'] and np.isnan(data['preferences']).all() and not meta['human_reviewed']
    # Synthetic origin is retained too; explicit preview-only status cannot be bypassed.
    with pytest.raises(ValueError,match='Preview-only'):SessionDataset(out.parent,'test',cfg,6,1,allow_synthetic=True)

def test_prepare_refuses_oversized_working_set_before_decoding(tmp_path,cfg):
    manifest=tmp_path/'session.json';manifest.write_text(json.dumps(dict(session_id='big',split='train',duration_seconds=120,alignment_verified=True,human_reviewed=False,cameras=[dict(id='a',video='missing.mp4',audio=None),dict(id='b',video='missing.mp4',audio=None)])))
    large=ModelConfig(max_cameras=2,image_size=512,audio_samples=64000,decision_hz=30,hidden_dim=16,layers=1,memory_tokens=4)
    with pytest.raises(ValueError,match='512MiB'):prepare(manifest,tmp_path/'never.npz',large)

def test_annotation_cannot_compare_keep_against_the_same_pgm(tmp_path):
    store=AnnotationStore(tmp_path);session=media_fixture(tmp_path);session['labels']=[dict(time_s=2,preferred='keep',rejected=['a'])]
    with pytest.raises(ValueError,match='同一个动作'):store.create(session)

def test_different_clips_of_same_source_video_cannot_cross_splits(tmp_path,cfg):
    create_fixture(tmp_path,cfg,steps=12,sessions=3)
    for name in ('train_0.npz','test_0.npz'):
        path=tmp_path/name
        with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        m=json.loads(str(values['metadata'].item()));m['source_media_sha256']={'camera_a':{'video':'a'*64}};values['metadata']=np.array(json.dumps(m));np.savez_compressed(path,**values)
    report=quality(tmp_path)
    assert not report['passed'] and any('同一源视频' in e for e in report['errors'])

def test_review_flags_must_not_be_truthy_strings(tmp_path,cfg):
    from camix_director.data import read_pack
    create_fixture(tmp_path,cfg,steps=12,sessions=3);p=tmp_path/'train_0.npz'
    with np.load(p,allow_pickle=False) as z:values={k:z[k] for k in z.files}
    m=json.loads(str(values['metadata'].item()));m['human_reviewed']='false';values['metadata']=np.array(json.dumps(m));np.savez_compressed(p,**values)
    with pytest.raises(ValueError,match='JSON boolean'):read_pack(p,cfg)
