import io,json
from pathlib import Path
import numpy as np
import pytest
from camix_director.ami_originals import BASE,CONFIG,validate_entry,fetch_entry,weak_preferences,disk_guard,ModelConfig
from camix_director.storage import digest_file


def entry(size=1000):return dict(meeting='ES2006a',split='train',agent='A',kind='video',name='ES2006a.Closeup2_orig.avi',url=BASE+'/ES2006a/video/ES2006a.Closeup2_orig.avi',bytes=size,etag='"fixed"')
class Response(io.BytesIO):
    def __init__(self,data,status=200,**headers):super().__init__(data);self.status=status;self.headers={'Content-Length':str(len(data)),'ETag':'"fixed"',**headers}


def test_originals_config_and_all_four_speaker_weak_labels():
    cfg=ModelConfig(**CONFIG);assert cfg.image_size==224 and cfg.max_cameras==4
    intervals=[[(0,1)],[(1,2)],[(2,3)],[(3,4)]]
    p=weak_preferences(intervals,np.array([.5,1.5,2.5,3.5,5.]),program=2)
    assert p[:4,1:].argmax(1).tolist()==[0,1,2,3]
    assert p[:4,0].tolist()==[0,0,1,0] and np.isnan(p[-1]).all()
    intervals[1].append((2,3))
    assert np.isnan(weak_preferences(intervals,np.array([2.5]),0)).all()


@pytest.mark.parametrize('key,value',[('name','../secret'),('url','https://attacker.invalid/x'),('split','test'),('kind','other'),('meeting','ES2002a')])
def test_source_path_and_split_are_fixed(key,value):
    e=entry();e[key]=value
    with pytest.raises(ValueError):validate_entry(e)


def test_sequential_fetch_hash_receipt_and_idempotence(tmp_path):
    blob=b'video'*200;calls=[]
    def open_(request,timeout):calls.append(request);return Response(blob)
    dest=tmp_path/'source.avi';e=entry(len(blob))
    result=fetch_entry(e,dest,reserve_bytes=0,opener=open_)
    assert dest.read_bytes()==blob and result['sha256']==digest_file(dest)
    assert result['publisher_checksum_available'] is False
    assert fetch_entry(e,dest,reserve_bytes=0,opener=open_)==result and len(calls)==1
    dest.write_bytes(b'bad')
    with pytest.raises(ValueError,match='changed'):fetch_entry(e,dest,reserve_bytes=0,opener=open_)


def test_resumes_exact_range_and_etag(tmp_path):
    blob=b'a'*1000;dest=tmp_path/'source.avi';e=entry()
    dest.with_suffix('.avi.part').write_bytes(blob[:300])
    dest.with_suffix('.avi.part.json').write_text(json.dumps({k:e.get(k) for k in ('url','bytes','etag')}))
    def open_(request,timeout):
        assert request.headers['Range']=='bytes=300-' and request.headers['If-range']=='"fixed"'
        return Response(blob[300:],status=206,**{'Content-Range':'bytes 300-999/1000'})
    fetch_entry(e,dest,reserve_bytes=0,opener=open_)
    assert dest.read_bytes()==blob and not dest.with_suffix('.avi.part').exists()


def test_ignored_resume_range_never_appends(tmp_path):
    dest=tmp_path/'source.avi';e=entry();part=dest.with_suffix('.avi.part');part.write_bytes(b'a'*300)
    dest.with_suffix('.avi.part.json').write_text(json.dumps({k:e.get(k) for k in ('url','bytes','etag')}))
    with pytest.raises(ValueError,match='exact resume'):
        fetch_entry(e,dest,reserve_bytes=0,opener=lambda req,timeout:Response(b'a'*1000))
    assert part.stat().st_size==300 and not dest.exists()


def test_source_etag_change_and_missing_receipt_refused(tmp_path):
    dest=tmp_path/'source.avi'
    with pytest.raises(ValueError,match='ETag'):
        fetch_entry(entry(),dest,reserve_bytes=0,opener=lambda req,timeout:Response(b'a'*1000,ETag='"new"'))
    dest.write_bytes(b'a'*1000)
    with pytest.raises(ValueError,match='without receipt'):fetch_entry(entry(),dest,reserve_bytes=0)


def test_disk_protection_and_cancel(tmp_path,monkeypatch):
    import camix_director.ami_originals as original
    monkeypatch.setattr(original.shutil,'disk_usage',lambda p:type('Disk',(),{'free':100})())
    with pytest.raises(OSError,match='free disk'):disk_guard(tmp_path,90,20)
    stop=tmp_path/'stop';stop.touch();dest=tmp_path/'video.avi'
    with pytest.raises(InterruptedError):fetch_entry(entry(),dest,reserve_bytes=0,stop_file=stop,opener=lambda req,timeout:Response(b'a'*1000))
    assert not dest.exists()


def test_browser_clip_keeps_source_resolution_and_interval(tmp_path):
    import av
    from fractions import Fraction
    from camix_director.ami_review import source_resolution_clip
    source=tmp_path/'source.mp4'
    with av.open(str(source),'w') as c:
        stream=c.add_stream('mpeg4',rate=25);stream.width=720;stream.height=576;stream.pix_fmt='yuv420p'
        for i in range(50):
            frame=av.VideoFrame.from_ndarray(np.full((576,720,3),i*3,np.uint8),format='rgb24')
            frame.pts=i;frame.time_base=Fraction(1,25)
            for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    dest=tmp_path/'review.mp4';count=source_resolution_clip(source,dest,.4,1)
    with av.open(str(dest)) as c:
        stream=c.streams.video[0]
        assert (stream.width,stream.height)==(720,576) and float(stream.average_rate)==25
        assert stream.codec_context.name=='h264'
        frames=list(c.decode(stream));assert len(frames)==count==25
        assert abs(float(frames[0].to_ndarray(format='rgb24').mean())-30)<8
    with pytest.raises(ValueError,match='overwrite'):source_resolution_clip(source,dest,.4,1)



def test_timeout_resumes_from_committed_bytes_without_redownload(tmp_path,monkeypatch):
    import camix_director.ami_originals as original
    monkeypatch.setattr(original.time,'sleep',lambda seconds:None)
    blob=b'a'*1000;calls=[]
    class Truncated(Response):
        def read(self,size=-1):
            if self.tell()>=300:raise TimeoutError('simulated interrupted network')
            return super().read(min(300,size))
    def open_(request,timeout):
        assert timeout==120
        calls.append(request)
        if len(calls)==1:return Truncated(blob)
        assert request.headers['Range']=='bytes=300-'
        return Response(blob[300:],status=206,**{'Content-Range':'bytes 300-999/1000'})
    dest=tmp_path/'source.avi';result=fetch_entry(entry(),dest,reserve_bytes=0,opener=open_)
    assert dest.read_bytes()==blob and len(calls)==2 and result['sha256']==digest_file(dest)


def test_exhausted_timeouts_are_bounded_and_preserve_partial(tmp_path,monkeypatch):
    import camix_director.ami_originals as original
    monkeypatch.setattr(original.time,'sleep',lambda seconds:None)
    calls=[]
    def offline(request,timeout):calls.append(request);raise TimeoutError('offline')
    dest=tmp_path/'source.avi'
    with pytest.raises(TimeoutError):fetch_entry(entry(),dest,reserve_bytes=0,opener=offline)
    assert len(calls)==12 and not dest.exists()
    assert dest.with_suffix('.avi.part.json').exists()


def test_main_persists_failure_instead_of_leaving_running_status(tmp_path,monkeypatch):
    import camix_director.ami_originals as original
    root=tmp_path/'data';root.mkdir();(root/'download-plan.json').write_text('{}')
    monkeypatch.setattr(original,'acquire',lambda *args,**kwargs:(_ for _ in ()).throw(TimeoutError('offline')))
    monkeypatch.setattr('sys.argv',['ami_originals','download','--root',str(root),'--accept-ami-cc-by'])
    with pytest.raises(TimeoutError):original.main()
    state=json.loads((root/'pipeline-status.json').read_text())
    assert state['status']=='failed' and state['phase']=='downloading' and 'offline' in state['error']
    assert state['training_started'] is False



def test_fresh_clone_uses_public_source_plan_template(tmp_path,monkeypatch):
    import camix_director.ami_originals as original
    received=[]
    def capture(root,plan,*args):
        received.append(plan)
        return {'source_count':len(plan['entries'])}
    root=tmp_path/'fresh-clone-data'
    monkeypatch.setattr(original,'acquire',capture)
    monkeypatch.setattr('sys.argv',['ami_originals','download','--root',str(root),'--accept-ami-cc-by'])
    original.main()
    assert len(received[0]['entries'])==32 and received[0]['total_bytes']==6300005744
    assert json.loads((root/'pipeline-status.json').read_text())['status']=='completed'
