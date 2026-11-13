import wave
import numpy as np
import pytest
from camix_director.config import ModelConfig
from camix_director.media import audio_samples,video_samples

def test_pcm_windows_use_past_samples(tmp_path):
    pytest.importorskip('av')
    rate=16000;values=(np.sin(np.arange(rate)*.03)*10000).astype('<i2')
    path=tmp_path/'test.wav'
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(values.tobytes())
    cfg=ModelConfig(max_cameras=2,image_size=16,audio_samples=256,hidden_dim=16,memory_tokens=4)
    times=np.array([.1,.2,.3])
    samples,valid=audio_samples(str(path),0,times,cfg)
    assert valid.tolist()==[1,1,1]
    for index,t in enumerate(times):
        end=round(t*rate)
        np.testing.assert_allclose(samples[index],values[end-256:end]/32768,atol=1e-6)

def test_video_uses_latest_past_frame(tmp_path):
    av=pytest.importorskip('av');path=tmp_path/'test.mp4'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=32;stream.height=32;stream.pix_fmt='yuv420p'
        for i in range(5):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i*40,np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    frames,valid,ages=video_samples(str(path),0,np.array([.15,.25,.35]),16)
    assert valid.tolist()==[1,1,1]
    np.testing.assert_allclose(ages,[.05,.05,.05],atol=.001)
    assert abs(float(frames[0].mean())-40)<8
    assert abs(float(frames[1].mean())-80)<8

def test_range_reader_seek_cache_and_bounded_reads(monkeypatch):
    import io
    from camix_director.media import HTTPRangeReader
    blob=bytes(range(256))*4;requests=[]
    class Response(io.BytesIO):
        status=206
        def __init__(self,data,headers):super().__init__(data);self.headers=headers
    def request(req,timeout):
        requests.append(req)
        if req.get_method()=='HEAD':return Response(b'',{'Content-Length':str(len(blob)),'ETag':'fixed'})
        start,end=map(int,req.headers['Range'].split('=')[1].split('-'))
        return Response(blob[start:end+1],{'Content-Range':f'bytes {start}-{end}/{len(blob)}'})
    monkeypatch.setattr('urllib.request.urlopen',request)
    reader=HTTPRangeReader('https://example.invalid/file',block_size=128,max_blocks=2)
    reader.seek(120);assert reader.read(24)==blob[120:144]
    count=len(requests);reader.seek(125);assert reader.read(8)==blob[125:133]
    assert len(requests)==count
    reader.seek(-8,2);assert reader.read(20)==blob[-8:]
    assert len(reader.cache)<=2
    reader.seek(0)
    with pytest.raises(ValueError,match='Unbounded'):reader.read()
