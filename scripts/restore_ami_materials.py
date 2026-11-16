"""Verify and move downloaded AMI release assets into a source checkout.

No network requests, no training and no overwrite of differing existing files.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024**2):h.update(block)
    return h.hexdigest()
def save_if_missing(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    tmp=path.with_suffix(path.suffix+'.restoring')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-dir',required=True);p.add_argument('--project',default='.')
    a=p.parse_args();download=Path(a.download_dir).resolve();project=Path(a.project).resolve()
    if not (project/'pyproject.toml').is_file():p.error('--project must be the cloned CamixDirectorTraining root')
    manifest=json.loads((download/'ami-materials-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format')!='camix_ami_release_materials_v1':raise ValueError('Unknown material manifest')
    if len(manifest.get('files',[]))!=48 or len({r['asset_name'] for r in manifest['files']})!=48:raise ValueError('Expected48 unique material files')
    if sum(r['kind']=='weak_training_pack' for r in manifest['files'])!=16:raise ValueError('Expected16 prepared packs')
    data=(project/'data/ami_originals_576').resolve();verified=set();missing=[]
    for row in manifest['files']:
        name=row['asset_name']
        if Path(name).name!=name:raise ValueError('Unsafe asset basename')
        destination=(project/row['relative']).resolve()
        if not destination.is_relative_to(data):raise ValueError('Asset destination escapes dataset')
        source=download/name
        if source.is_symlink():raise ValueError('Refuse symlink download')
        if destination.exists():
            if destination.is_symlink() or destination.stat().st_size!=row['bytes'] or sha(destination)!=row['sha256']:
                raise ValueError('Existing destination differs; preserved without overwrite: '+str(destination))
            verified.add(row['relative']);continue
        if not source.is_file():missing.append(name);continue
        if source.stat().st_size!=row['bytes'] or sha(source)!=row['sha256']:raise ValueError('Download checksum mismatch: '+name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.stat().st_dev!=destination.parent.stat().st_dev:
            if shutil.disk_usage(destination.parent).free<row['bytes']+1024**3:raise OSError('Not enough space to move across disks safely')
            temp=destination.with_suffix(destination.suffix+'.restoring');shutil.copyfile(source,temp)
            if sha(temp)!=row['sha256']:raise ValueError('Cross-disk copy mismatch')
            temp.replace(destination);source.unlink()
        else:source.replace(destination)
        verified.add(row['relative']);print('Restored',row['relative'])
    media=[r for r in manifest['files'] if r['kind']!='weak_training_pack']
    packs=[r for r in manifest['files'] if r['kind']=='weak_training_pack']
    raw_ready=all(r['relative'] in verified for r in media)
    packs_ready=all(r['relative'] in verified for r in packs)
    if raw_ready:
        save_if_missing(data/'source-manifest.json',manifest['source_manifest'])
        save_if_missing(data/'download-plan.json',manifest['download_plan'])
    if packs_ready:save_if_missing(data/'preparation-report.json',manifest['preparation_report'])
    for src,dest in [('AMI-ATTRIBUTION.md','ATTRIBUTION.md'),('CC-BY-4.0.txt','LICENSE-CC-BY-4.0.txt'),('AMI-DATASET-README-zh.md','RELEASE_MATERIALS_README.md')]:
        source=download/src;target=data/dest
        if source.exists() and not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    print(json.dumps({'raw_media_ready':raw_ready,'prepared_v1_ready':packs_ready,'verified_files':len(verified),'missing_files':len(missing),'training_started':False},ensure_ascii=False))
    if not raw_ready and not packs_ready:raise SystemExit('No complete raw or prepared subset yet; download missing assets and rerun')
if __name__=='__main__':main()
