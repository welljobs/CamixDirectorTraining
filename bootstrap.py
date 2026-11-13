#!/usr/bin/env python3
"""Fetch compatible official wheels with checksums; install offline into .venv.
No global Python/brew edits. Run with the intended virtualenv Python 3.11.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess,sys,urllib.request
from pathlib import Path
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename
from camix_director.download import download

ROOT=Path(__file__).resolve().parent

def wheel(package,version):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json",timeout=30) as response:meta=json.load(response)
    tags={tag:i for i,tag in enumerate(sys_tags())};choices=[]
    for file in meta['urls']:
        if file['packagetype']!='bdist_wheel':continue
        try:_,_,_,wheel_tags=parse_wheel_filename(file['filename'])
        except ValueError:continue
        compatible=[tags[x] for x in wheel_tags if x in tags]
        if compatible:choices.append((min(compatible),file))
    if not choices:raise RuntimeError(f"No binary wheel for {package}=={version} on this interpreter")
    file=min(choices,key=lambda x:x[0])[1]
    url=file['url']
    if package.lower()=='torch' and file['filename']=='torch-2.7.0-cp311-none-macosx_11_0_arm64.whl':
        url='https://download.pytorch.org/whl/cpu/'+file['filename']
    result=download(url,ROOT/'downloads'/file['filename'],sha256=file['digests']['sha256'],workers=6)
    result.update(package=package,version=version);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--download-only',action='store_true');a=p.parse_args()
    if not ((3,11)<=sys.version_info[:2]<(3,13)):raise SystemExit('Use Python 3.11 (recommended) or compatible 3.12')
    if sys.platform=='win32':raise SystemExit('Use setup_windows.bat; the macOS lock includes Core ML. See WINDOWS.md')
    lock=[line.strip().split('==',1) for line in (ROOT/'requirements.lock').read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith('#')]
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda x:wheel(*x),lock))
    (ROOT/'downloads'/'wheel-sources.json').write_text(json.dumps(results,indent=2), encoding="utf-8")
    if not a.download_only:
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--find-links',str(ROOT/'downloads'),'-r',str(ROOT/'requirements.lock')],check=True)
        subprocess.run([sys.executable,'-m','pip','check'],check=True)

if __name__=='__main__':main()
