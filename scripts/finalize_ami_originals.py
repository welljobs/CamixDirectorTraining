"""One-off post-acquisition quality/review task. Does not train or control Camix."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
workspace=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(workspace))
from camix_director.dataset_tools import quality
from camix_director.ami_review import create
from camix_director.runtime import atomic_json
root=workspace/'data/ami_originals_576';reports=workspace/'reports/ami-originals-576'
if a.wait_pid:
    deadline=time.monotonic()+48*3600
    while not (root/'preparation-report.json').exists():
        if time.monotonic()>deadline:raise SystemExit('Acquisition wait timed out; no training started')
        result=subprocess.run(['ps','-p',str(a.wait_pid),'-o','args='],capture_output=True,text=True)
        if result.returncode or '-m camix_director.ami_originals download' not in result.stdout:raise SystemExit('Acquisition exited before preparation success; inspect pipeline log')
        time.sleep(15)
prepared=json.loads((root/'preparation-report.json').read_text())
print('Checking all prepared data and original-event split separation...',flush=True)
result=quality(prepared['prepared_directory'],cache_root=workspace/'.cache/sequences')
atomic_json(reports/'quality-report.json',result)
if not result['passed']:raise SystemExit('Quality check failed; no review projects or training started')
print('Quality PASSED. Creating unreviewed source-resolution review projects...',flush=True)
review=create(workspace,root)
atomic_json(reports/'review-report.json',review)
atomic_json(reports/'acceptance.json',dict(source_manifest='data/ami_originals_576/source-manifest.json',source_bytes=prepared['original_sources_bytes'],prepared_directory=str(Path(prepared['prepared_directory']).relative_to(workspace)),quality_passed=result['passed'],original_sessions=result['sessions'],prepared_episodes=len(prepared['episodes']),review_projects=len(review['projects']),model_training_started=False,production_approved=False))
print('Acquisition / preparation / quality / browser review complete. No model was automatically trained or replaced.',flush=True)
