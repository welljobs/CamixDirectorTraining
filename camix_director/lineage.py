"""Local, auditable experiment lineage; never promotes models to production."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime,timezone
from .runtime import atomic_json
from .storage import digest_file


def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def environment_snapshot():
    import torch
    packages={d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()}
    return dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),torch=str(torch.__version__),cuda_runtime=torch.version.cuda,packages=dict(sorted(packages.items())))


def code_snapshot():
    root=Path(__file__).parent
    return {p.name:digest_file(p) for p in sorted(root.glob('*.py'))}


def dataset_identity(provenance):
    entries=[{k:p.get(k) for k in ('session_id','split','pack_sha256','annotation_version','review')} for p in provenance]
    return canonical_hash(sorted(entries,key=lambda p:(p.get('session_id') or '',p.get('split') or '',p.get('pack_sha256') or '')))


def start_run(output,config,settings,provenance,parent=None):
    value=dict(format='camix_run_manifest_v2',created_at=datetime.now(timezone.utc).isoformat(),
        config=config.to_dict(),settings=settings,data=provenance,dataset_fingerprint=dataset_identity(provenance),
        code=code_snapshot(),environment=environment_snapshot(),parent=parent,production_approved=False)
    snapshot=Path(output)/'source_snapshot/camix_director';snapshot.mkdir(parents=True,exist_ok=True)
    import shutil
    for source in Path(__file__).parent.glob('*.py'):shutil.copy2(source,snapshot/source.name)
    atomic_json(Path(output)/'run-manifest.json',value)
    return value


def model_card(output,checkpoint,training,qualification):
    output=Path(output)
    payload=dict(format='camix_model_card_v1',checkpoint_sha256=digest_file(checkpoint),qualification=qualification,
        production_approved=False,status='research_only',training=training,
        known_limits=['Not director-ground-truth unless independently reviewed','No independent audio routing head','Numerical export parity is not device performance or director quality','Closed-loop acceptance and multi-session human review required'],
        lineage_file='run-manifest.json')
    atomic_json(output/'model-card.json',payload)
    return payload
