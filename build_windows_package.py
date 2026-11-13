#!/usr/bin/env python3
"""Build a portable source/data package, never a copied macOS environment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def build(output,include_data=False):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Choose a new ZIP name; never overwrite an existing delivery')
    files=set()
    for pattern in ('*.py','*.md','*.toml','requirements*.txt','requirements*.lock','*.bat','*.command','.gitignore'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('camix_director','tests','examples','docs','.github/workflows'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':files.add(path)
    for name in ('RESULTS.md','summary.json','web-console-verification.json'):
        if (ROOT/'reports'/name).exists():files.add(ROOT/'reports'/name)
    if include_data:
        for folder in ('ami_pilot','ami_fresh_holdout'):
            parent=ROOT/'data'/folder
            files.update((parent/'prepared').glob('*.npz'))
            if (parent/'source-report.json').is_file():files.add(parent/'source-report.json')
            for filename in ('ami-license-page.html','ami-signals.html'):
                if (parent/'sources'/filename).is_file():files.add(parent/'sources'/filename)
        run=ROOT/'runs/ami_normalized_balanced_20261112'
        for name in ('best.pt','training-report.json','evaluation_fresh/evaluation.json'):
            if (run/name).is_file():files.add(run/name)
    for path in files:
        if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Package entry escapes project')
    total=sum(p.stat().st_size for p in files)
    if total>512*1024*1024:raise ValueError('Portable bundle exceeds512MiB; use a source-only package')
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),format='portable_source_NOT_a_Windows_executable',
        windows_hardware_tested=False,includes_public_pilot_data=include_data,production_approved=False,
        excluded=['.venv','.venv-win','.web','downloads','arbitrary_user_media'],
        files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in sorted(files):archive.write(path,'CamixDirectorTraining/'+path.relative_to(ROOT).as_posix())
            archive.writestr('CamixDirectorTraining/PORTABLE_PACKAGE.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    except Exception:
        output.unlink(missing_ok=True);raise
    return dict(path=str(output),files=len(files),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--include-data',action='store_true');args=parser.parse_args()
    print(json.dumps(build(args.output,args.include_data),indent=2))
