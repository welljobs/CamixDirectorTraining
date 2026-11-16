"""Mirror only approved public AMI signals to an existing dataset Release draft.

Prepared packs must already be uploaded separately. No training, paid runner,
private data, repository clone data, source overwrite or asset clobber occurs.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,re,shutil,subprocess,tempfile,time,urllib.request,threading

REPOSITORY='welljobs/CamixDirectorTraining'
BASE='https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()
def gh(*args,as_json=False,timeout=120):
    r=subprocess.run(['gh',*args],capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError('GitHub CLI operation failed: '+r.stderr[-1200:])
    return json.loads(r.stdout) if as_json else r.stdout

def validate(plan):
    if plan.get('format')!='camix_ami_release_publish_plan_v1' or plan.get('repository')!=REPOSITORY:raise ValueError('Unexpected release plan')
    raw=plan.get('raw',[]);prepared=plan.get('prepared',[])
    if len(raw)!=32 or len(prepared)!=16 or len({r['asset_name'] for r in raw+prepared})!=48:raise ValueError('Expected32 public signals and16 prepared packs')
    for r in raw:
        if r['meeting'] not in ('ES2006a','ES2007a','ES2008a','ES2009a') or r['kind'] not in ('video','audio'):raise ValueError('Unexpected AMI signal')
        pattern=re.escape(r['meeting'])+(r'\.Closeup[1-4]_orig\.avi' if r['kind']=='video' else r'\.Headset-[0-3]\.wav')
        if not re.fullmatch(pattern,r['asset_name']):raise ValueError('Unexpected signal filename')
        if r['url']!=f"{BASE}/{r['meeting']}/{r['kind']}/{r['asset_name']}":raise ValueError('Only exact official AMI URLs are permitted')
        if not 0<r['bytes']<512*1024**2 or not re.fullmatch('[a-f0-9]{64}',r['sha256']):raise ValueError('Invalid size/SHA256')
    if sum(r['bytes'] for r in raw)!=6300005744 or sum(r['bytes'] for r in prepared)!=477260891:raise ValueError('Material budget changed')
    if not isinstance(plan.get('release_id'),int) or not re.fullmatch('ami-originals-576-[0-9]{8}',plan['tag']):raise ValueError('Unsafe release identity')

def match(asset,expected):
    return asset['state']=='uploaded' and asset['size']==expected['bytes'] and asset.get('digest')=='sha256:'+expected['sha256']

NETWORK_SLOTS=threading.Semaphore(3)

def download(row,path,rate_mib=2):
    """Verified finite4MiB ranges; all workers share3 network slots (<=6MiB/s).

    A slow whole-file HTTP response cannot hold up an entire meeting. Completed
    prefix bytes are retained; only this runner's own range parts are removed.
    """
    expected=row['bytes'];offset=path.stat().st_size if path.exists() else 0
    if not 0<=offset<=expected:raise ValueError('Invalid source prefix')
    if offset==expected:
        if sha(path)!=row['sha256']:raise ValueError('Source checksum mismatch')
        return
    directory=path.parent/(path.name+'.ranges');directory.mkdir(exist_ok=True)
    length=4*1024**2
    segments=[(i,min(expected,i+length)-1) for i in range(offset,expected,length)]
    def part(segment):
        lo,hi=segment;target=directory/f'{lo}-{hi}';count=hi-lo+1
        if target.exists() and target.stat().st_size==count:return target
        for attempt in range(10):
            try:
                with NETWORK_SLOTS:
                    started=time.monotonic();received=0
                    headers={'Range':f'bytes={lo}-{hi}','Cache-Control':'no-cache','User-Agent':'CamixDirectorTraining verified AMI bounded-range mirror'}
                    if row.get('etag'):headers['If-Range']=row['etag']
                    with urllib.request.urlopen(urllib.request.Request(row['url'],headers=headers),timeout=60) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{expected}':raise ValueError('Exact finite range required')
                        if int(response.headers.get('Content-Length','-1'))!=count:raise ValueError('Range size changed')
                        if row.get('etag') and response.headers.get('ETag') and response.headers['ETag']!=row['etag']:raise ValueError('Source ETag changed')
                        temporary=target.with_suffix('.partial')
                        with temporary.open('wb') as output:
                            while True:
                                block=response.read(min(64*1024,count-received+1))
                                if not block:break
                                received+=len(block)
                                if received>count:raise ValueError('Overlong range')
                                output.write(block)
                                delay=received/(rate_mib*1024**2)-(time.monotonic()-started)
                                if delay>0:time.sleep(delay)
                        if received!=count:raise OSError('Truncated range')
                    temporary.replace(target)
                return target
            except ValueError:raise
            except Exception as error:
                print('RANGE RETRY',row['asset_name'],lo,attempt+1,type(error).__name__,flush=True)
                if attempt==9:raise
                time.sleep(min(2**attempt,15))
    with ThreadPoolExecutor(max_workers=3) as pool:
        with path.open('ab') as output:
            for segment,chunk in zip(segments,pool.map(part,segments)):
                if output.tell()!=segment[0]:raise ValueError('Noncontiguous recovered source')
                with chunk.open('rb') as source:shutil.copyfileobj(source,output,1024**2)
                output.flush();chunk.unlink()
    directory.rmdir()
    if path.stat().st_size!=expected or sha(path)!=row['sha256']:raise ValueError('Official signal checksum mismatch; no upload')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text(encoding='utf-8'));validate(plan)
    repo=plan['repository'];rid=plan['release_id'];tag=plan['tag']
    if os.environ.get('GITHUB_REPOSITORY',repo)!=repo:raise ValueError('Workflow repository differs from publish plan')
    actual=gh('api',f'repos/{repo}/releases/{rid}',as_json=True)
    if actual['tag_name']!=tag or actual['target_commitish']!=plan['code_commit']:raise ValueError('Release target changed')
    if not actual['draft']:raise ValueError('Release already public; refusing to alter it')
    def assets():return gh('api',f'repos/{repo}/releases/{rid}/assets?per_page=100',as_json=True)
    def existing(name):return next((x for x in assets() if x['name']==name),None)
    with tempfile.TemporaryDirectory(prefix='ami-signal-release-') as temp:
        directory=Path(temp)
        def mirror(row):
            found=existing(row['asset_name'])
            if found:
                if not match(found,row):raise ValueError('Conflicting asset; never clobber')
                print('REUSED',row['asset_name'],flush=True);return
            if shutil.disk_usage(directory).free<2*1024**3:raise OSError('Runner disk below2GiB free')
            path=directory/row['asset_name'];print('DOWNLOAD',row['asset_name'],row['bytes'],flush=True)
            download(row,path)
            for attempt in range(3):
                try:gh('release','upload',tag,str(path),'--repo',repo,timeout=3600)
                except Exception:
                    if attempt==2:raise
                found=existing(row['asset_name'])
                if found and match(found,row):break
                if found:raise ValueError('Incomplete/conflicting upload retained; do not clobber')
                if attempt==2:raise ValueError('Upload not confirmed')
                time.sleep(5)
            path.unlink();print('SHA256 VERIFIED',row['asset_name'],flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(mirror,plan['raw']))
        # Wait for the user's local derived packs; never silently regenerate them
        # on another platform and pretend their bytes match the original packs.
        expected=plan['prepared']+plan['raw']+[{**r,'asset_name':r['name']} for r in plan['fixed_assets']]
        for wait in range(60):
            remote={r['name']:r for r in assets()}
            missing=[r['asset_name'] for r in expected if r['asset_name'] not in remote]
            if not missing:break
            print('WAITING FOR PREPARED/DOCUMENT ASSETS',len(missing),flush=True);time.sleep(30)
        else:raise ValueError('Prepared/doc assets still missing; keep draft and rerun later')
        for row in expected:
            if not match(remote[row['asset_name']],row):raise ValueError('Final expected asset digest mismatch')
        def published_entry(row):
            x=remote[row['asset_name']]
            return dict(name=x['name'],bytes=x['size'],sha256=row['sha256'],kind=row['kind'],id=x['id'],url=x['browser_download_url'],remote_digest=x['digest'])
        material_assets=[published_entry(r) for r in plan['prepared']+plan['raw']]
        fixed=[published_entry({**r,'asset_name':r['name']}) for r in plan['fixed_assets']]
        files=[{k:v for k,v in r.items() if k not in ('url','etag')} for r in plan['raw']+plan['prepared']]
        manifest=dict(format='camix_ami_release_materials_v1',created_at=datetime.now(timezone.utc).isoformat(),tag=tag,code_commit=plan['code_commit'],repository=repo,release_url=plan['release_url'],
            original_media_files=32,original_media_bytes=6300005744,prepared_pack_files=16,prepared_pack_bytes=477260891,
            license='CC BY4.0 for AMI signals/transcription and derived weak packs; not a software license',human_reviewed=False,professional_director_ground_truth=False,prepared_schema_version=1,v2_reviewed_corpus=False,production_approved=False,
            assets=material_assets+fixed,files=files,source_manifest=plan['source_manifest'],download_plan=plan['download_plan'],preparation_report=plan['preparation_report'])
        manifest_path=directory/'ami-materials-manifest.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        prior=existing(manifest_path.name)
        if prior:raise ValueError('Manifest already present; inspect before changing a completed publication')
        gh('release','upload',tag,str(manifest_path),'--repo',repo)
        mrow=existing(manifest_path.name);mexpected=dict(bytes=manifest_path.stat().st_size,sha256=sha(manifest_path))
        if not mrow or not match(mrow,mexpected):raise ValueError('Manifest upload hash mismatch')
        allchecks=material_assets+fixed+[dict(name=manifest_path.name,sha256=mexpected['sha256'])]
        sums=directory/'SHA256SUMS';sums.write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in sorted(allchecks,key=lambda r:r['name'])))
        gh('release','upload',tag,str(sums),'--repo',repo)
        srow=existing(sums.name)
        if not srow or not match(srow,dict(bytes=sums.stat().st_size,sha256=sha(sums))):raise ValueError('Checksums upload hash mismatch')
        final=assets()
        expected_names={r['asset_name'] for r in expected}|{manifest_path.name,sums.name}
        if len(final)!=54 or {r['name'] for r in final}!=expected_names:raise ValueError('Unexpected or incomplete release')
        if a.publish:
            request=directory/'publish.json';request.write_text(json.dumps(dict(draft=False,prerelease=True,make_latest='false',body=plan['release_notes'])))
            gh('api','--method','PATCH',f'repos/{repo}/releases/{rid}','--input',str(request))
            final=gh('api',f'repos/{repo}/releases/{rid}',as_json=True)
            if final['draft']:raise ValueError('Publication not confirmed')
            print('PUBLISHED',final['html_url'],flush=True)
        else:print('All54 assets verified; draft preserved (publish not requested)',flush=True)
if __name__=='__main__':main()
