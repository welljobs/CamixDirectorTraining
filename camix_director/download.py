"""Resumable bounded downloads; TLS validation stays enabled; verify hashes when known."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,math,os,time,urllib.request
from pathlib import Path


def _download_unlocked(url,destination,max_bytes=512*1024*1024,sha256=None,workers=6):
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    headers={"User-Agent":"CamixDirectorTraining/0.1 (dataset reproducibility)"}
    with urllib.request.urlopen(urllib.request.Request(url,method="HEAD",headers=headers),timeout=30) as r:
        size=int(r.headers.get("Content-Length","0")); ranges=r.headers.get("Accept-Ranges","").lower()=="bytes"
    if not 0<size<=max_bytes:raise ValueError(f"Missing/excessive content length: {size} for {url}")
    if dest.exists() and dest.stat().st_size==size:
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        if sha256 is None or digest==sha256:return {"path":str(dest),"bytes":size,"sha256":digest,"url":url}
    partdir=dest.parent/(dest.name+".parts");partdir.mkdir(exist_ok=True)
    chunk=2*1024*1024 if ranges else size
    jobs=[(i,start,min(start+chunk,size)-1) for i,start in enumerate(range(0,size,chunk))]
    def fetch(job):
        i,start,end=job;path=partdir/str(i);expected=end-start+1
        if path.exists() and path.stat().st_size==expected:return path
        for attempt in range(4):
            try:
                request_headers=dict(headers)
                if ranges:request_headers["Range"]=f"bytes={start}-{end}"
                with urllib.request.urlopen(urllib.request.Request(url,headers=request_headers),timeout=45) as r:
                    if ranges and (r.status!=206 or not r.headers.get("Content-Range","").startswith(f"bytes {start}-{end}/")):
                        raise ValueError("Server ignored or altered byte range")
                    tmp=path.with_suffix(".partial");count=0
                    with tmp.open("wb") as out:
                        while True:
                            data=r.read(min(65536,expected-count+1))
                            if not data:break
                            count+=len(data)
                            if count>expected:raise ValueError("Overlong download")
                            out.write(data)
                    if count!=expected:raise IOError(f"Truncated range {count}/{expected}")
                tmp.replace(path);return path
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    print(f"DOWNLOAD {dest.name} {size/1048576:.1f} MiB ranges={ranges}",flush=True)
    with ThreadPoolExecutor(max_workers=workers if ranges else 1) as pool:parts=list(pool.map(fetch,jobs))
    temp=dest.with_suffix(dest.suffix+".partial");digest=hashlib.sha256()
    with temp.open("wb") as out:
        for path in parts:
            with path.open("rb") as source:
                while data:=source.read(1024*1024):out.write(data);digest.update(data)
    if sha256 and digest.hexdigest()!=sha256:raise ValueError("SHA256 mismatch; final file NOT accepted")
    temp.replace(dest)
    for path in parts:path.unlink()
    partdir.rmdir()
    return {"path":str(dest),"bytes":size,"sha256":digest.hexdigest(),"url":url}

def download(url,destination,max_bytes=512*1024*1024,sha256=None,workers=6):
    from .runtime import file_lock
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with file_lock(destination.with_suffix(destination.suffix+".download.lock")):
        return _download_unlocked(url,destination,max_bytes,sha256,workers)

if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument("url");p.add_argument("destination");p.add_argument("--sha256")
    a=p.parse_args();print(json.dumps(download(a.url,a.destination,sha256=a.sha256),indent=2))
