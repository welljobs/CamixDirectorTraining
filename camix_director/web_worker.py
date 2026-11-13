"""Durable isolated compute worker with heartbeat and a lifetime workspace lock."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback
import uuid
from .runtime import atomic_json,file_lock
from .web_core import read_json,stamp

ALLOWED={'camix_director.'+name for name in ('train','evaluate','prepare','ami','export','synthetic','inspect_data','dataset_tools','replay','raw_demo','gates','media_proxy','v2')}


def run(directory):
    directory=Path(directory).resolve();spec=read_json(directory/'spec.json')
    if not spec or spec.get('module') not in ALLOWED:raise ValueError('Invalid worker module')
    state={k:v for k,v in spec.items() if k!='args'};exit_code=0;finished=threading.Event();identity=uuid.uuid4().hex
    def heartbeat():
        while not finished.is_set():
            atomic_json(directory/'heartbeat.json',dict(pid=os.getpid(),instance=identity,timestamp=time.time()))
            finished.wait(2)
    thread=None
    try:
        with file_lock(Path(spec['workspace'])/'.web/worker.lock',timeout=0):
            state.update(status='running',started_at=stamp(),worker_instance=identity);atomic_json(directory/'job.json',state)
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
            if (directory/'stop.request').exists():raise SystemExit(130)
            print('LOCAL EXPERIMENT ONLY. production_approved=false; no Camix control.',flush=True)
            sys.argv=[spec['module']]+spec['args']
            try:runpy.run_module(spec['module'],run_name='__main__',alter_sys=True)
            except SystemExit as error:
                if error.code not in (None,0):raise
            if (directory/'stop.request').exists():raise SystemExit(130)
            state['status']='completed'
    except (SystemExit,InterruptedError) as error:
        exit_code=130 if isinstance(error,InterruptedError) or (directory/'stop.request').exists() else error.code if isinstance(error.code,int) else 1
        state.update(status='cancelled' if exit_code==130 else 'failed',error='任务已停止，已有完整检查点保留' if exit_code==130 else str(error))
    except Exception as error:
        exit_code=1;state.update(status='failed',error=str(error));traceback.print_exc()
    finally:
        finished.set()
        if thread:thread.join(timeout=3)
        state.update(exit_code=exit_code,finished_at=stamp());atomic_json(directory/'job.json',state)
    return exit_code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job-dir',required=True)
    raise SystemExit(run(parser.parse_args().job_dir))
