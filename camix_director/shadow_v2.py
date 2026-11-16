"""Local feature-packet boundary and recorded shadow suggestions; no network/TAKE."""
from pathlib import Path
import json
import time
import numpy as np

from .annotations_v2 import bind_media
from .contracts_v2 import canonical, data_path
from .features_v2 import cache_header, iter_features
from .policy_io_v2 import checkpoint_hash
from .policy_replay_v2 import feature_packet
from .policy_runtime_v2 import StreamingDirector
from .prepare_v2 import validate_index
from .runtime import atomic_json
from .storage import digest_file


def packets_from_event(root,index,cache_id,output):
    validate_index(index);m=index['document']['manifest'];bind_media(root,m,verify=True)
    header=cache_header(root,cache_id)
    if header['identity']['index_sha256']!=index['index_sha256']:raise ValueError('Packet export index/cache mismatch')
    path=data_path(root,str(output),exists=False)
    if path.exists():raise ValueError('Packet output exists')
    path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.partial')
    current=m['initial_program'];last_cut=0.;position=0
    with partial.open('w',encoding='utf-8') as out:
        for chunk,arrays in iter_features(root,cache_id):
            for local,t in enumerate(arrays['times']):
                while position<len(m['program_events']) and m['program_events'][position]['time_s']<t:
                    cut=m['program_events'][position]
                    if cut['camera_id']!=current:current,last_cut=cut['camera_id'],cut['time_s']
                    position+=1
                packet=feature_packet(index,header,arrays,local,current,float(t)-last_cut,index['media_identity'])
                out.write(canonical(packet)+'\n')
    bind_media(root,m,verify=True);partial.replace(path)
    return dict(path=path.relative_to(Path(root).resolve()).as_posix(),sha256=digest_file(path),packets=index['ticks'],
                availability_source='declared_offline_offsets_NOT_live_RTC_capture',execute_allowed=False)


def _strict_line(raw):
    def invalid(v):raise ValueError('Non-finite packet JSON')
    def unique(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate packet JSON key')
            result[key]=value
        return result
    return json.loads(raw,parse_constant=invalid,object_pairs_hook=unique)


def run_shadow(root,checkpoint,input_path,output,min_hold_s=2.,min_probability=0.,stop_file=None):
    source=data_path(root,str(input_path));actor=StreamingDirector(checkpoint,min_hold_s,min_probability)
    source_hash=digest_file(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Choose empty recorded shadow output')
    latencies=[];count=0;started=time.monotonic();partial=output/'suggestions.partial'
    with source.open('rb') as packets,partial.open('w',encoding='utf-8') as out:
        while True:
            raw=packets.readline(4*1024*1024+1)
            if not raw:break
            if len(raw)>4*1024*1024:raise ValueError('Oversized feature packet')
            if stop_file and Path(stop_file).exists():raise InterruptedError('Recorded shadow stopped; no final acceptance report')
            packet=_strict_line(raw)
            begin=time.perf_counter();result=actor.step(packet);elapsed=(time.perf_counter()-begin)*1000
            if len(latencies)<10000:latencies.append(elapsed)
            out.write(canonical(result)+'\n');count+=1
    if not count:raise ValueError('No shadow packets')
    if digest_file(source)!=source_hash:raise ValueError('Shadow packet file changed during run')
    path=output/'suggestions.jsonl';partial.replace(path)
    report=dict(format='camix_recorded_shadow_v2',checkpoint_sha256=checkpoint_hash(checkpoint),feature_space_id=actor.checkpoint['feature_space_id'],
                input_sha256=source_hash,suggestions_sha256=digest_file(path),packets=count,elapsed_s=time.monotonic()-started,
                cpu_adapter_step_ms_p50=float(np.percentile(latencies,50)),cpu_adapter_step_ms_p95=float(np.percentile(latencies,95)),
                timing_sample='first_10000_steps_only; excludes_capture_network_decode_and_encoder',
                live_rtc_verified=False,probability_calibrated=False,qualification=actor.checkpoint['qualification'],
                production_approved=False,execute_allowed=False,
                warning='Recorded feature-packet shadow only; no real RTC source adapter, no mixer control, no target-device SLO acceptance')
    atomic_json(output/'shadow-report.json',report);return report
