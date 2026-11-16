from pathlib import Path
import hashlib,importlib.util,io,json,copy
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('material_publisher',ROOT/'scripts/publish_ami_materials.py')
publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)

def plan():return json.loads((ROOT/'examples/ami-materials-release-20261116.json').read_text())
def test_only_expected_official_material_budget_allowed():
 p=plan();publisher.validate(p)
 assert sum(r['bytes'] for r in p['raw'])==6300005744
 assert sum(r['bytes'] for r in p['prepared'])==477260891
@pytest.mark.parametrize('field,value',[('url','https://attacker.invalid/private'),('asset_name','../secret'),('sha256','not-a-sha'),('bytes',9999999999)])
def test_mirror_rejects_altered_source_identity(field,value):
 p=plan();p['raw'][0][field]=value
 with pytest.raises(ValueError):publisher.validate(p)
class Response(io.BytesIO):
 def __init__(self,data,status,headers):super().__init__(data);self.status=status;self.headers=headers

def test_download_resumes_exact_bytes_and_checks_sha(tmp_path,monkeypatch):
 blob=b'x'*1000;dest=tmp_path/'material';dest.write_bytes(blob[:300]);calls=[]
 row=dict(url='https://example.invalid/file',bytes=1000,etag='fixed',sha256=hashlib.sha256(blob).hexdigest(),asset_name='test')
 def open_(request,timeout):
  calls.append(request);assert request.headers['Range']=='bytes=300-999'
  return Response(blob[300:],206,{'Content-Length':'700','Content-Range':'bytes 300-999/1000'})
 monkeypatch.setattr(publisher.urllib.request,'urlopen',open_)
 publisher.download(row,dest)
 assert dest.read_bytes()==blob and len(calls)==1

def test_download_never_accepts_ignored_resume_range(tmp_path,monkeypatch):
 dest=tmp_path/'material';dest.write_bytes(b'x'*300)
 row=dict(url='https://example.invalid/file',bytes=1000,etag='fixed',sha256='0'*64,asset_name='test')
 monkeypatch.setattr(publisher.urllib.request,'urlopen',lambda *a,**k:Response(b'y'*1000,200,{'Content-Length':'1000'}))
 with pytest.raises(ValueError):publisher.download(row,dest)
 assert dest.read_bytes()==b'x'*300

def test_remote_server_digest_is_required():
 expected=dict(bytes=10,sha256='a'*64)
 assert publisher.match(dict(state='uploaded',size=10,digest='sha256:'+'a'*64),expected)
 assert not publisher.match(dict(state='uploaded',size=10,digest=None),expected)
 assert not publisher.match(dict(state='starter',size=0,digest=None),expected)

def test_bounded_ranges_reassemble_large_file_without_losing_prefix(tmp_path,monkeypatch):
 blob=bytes(range(251))*40000;dest=tmp_path/'material';dest.write_bytes(blob[:13]);requests=[]
 row=dict(url='https://example.invalid/file',bytes=len(blob),etag='fixed',sha256=hashlib.sha256(blob).hexdigest(),asset_name='test')
 def open_(request,timeout):
  bounds=request.headers['Range'].removeprefix('bytes=');lo,hi=map(int,bounds.split('-'))
  requests.append((lo,hi));assert hi-lo+1<=4*1024**2
  return Response(blob[lo:hi+1],206,{'Content-Length':str(hi-lo+1),'Content-Range':f'bytes {lo}-{hi}/{len(blob)}','ETag':'fixed'})
 monkeypatch.setattr(publisher.urllib.request,'urlopen',open_)
 publisher.download(row,dest,rate_mib=100000)
 assert dest.read_bytes()==blob and len(requests)==3
 assert min(lo for lo,hi in requests)==13

def test_range_content_corruption_is_not_accepted(tmp_path,monkeypatch):
 dest=tmp_path/'material';row=dict(url='https://example.invalid/file',bytes=1000,etag='fixed',sha256=hashlib.sha256(b'x'*1000).hexdigest(),asset_name='test')
 monkeypatch.setattr(publisher.urllib.request,'urlopen',lambda *a,**k:Response(b'y'*1000,206,{'Content-Length':'1000','Content-Range':'bytes 0-999/1000','ETag':'fixed'}))
 with pytest.raises(ValueError,match='checksum'):publisher.download(row,dest,rate_mib=100000)
