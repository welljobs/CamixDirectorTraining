"""Import a verified OFFLINE policy path for new expert corrections, never auto-label it."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np

from .annotations_v2 import EventStore, bind_media
from .contracts_v2 import load_json
from .features_v2 import iter_features,cache_header
from .prepare_v2 import validate_index
from .storage import digest_file
from .v2 import artifact_path


def create_correction(root,index,replay_report):
    validate_index(index)
    path=artifact_path(root,str(replay_report));report=load_json(path)
    if (report.get('format')!='camix_policy_replay_v2' or report.get('policy')!='learned'
            or report.get('index_sha256')!=index['index_sha256']
            or report.get('source_content_sha256')!=index['document']['content_sha256']
            or report.get('media_identity')!=index['media_identity'] or report.get('execute_allowed') is not False
            or not report.get('variant')):
        raise ValueError('Correction requires the exact untruncated learned replay of this event revision')
    if cache_header(root,report['cache_id'])['identity']['index_sha256']!=index['index_sha256']:
        raise ValueError('Correction feature cache belongs to another event')
    trace=path.parent/'value-trace.jsonl'
    if not trace.is_file() or digest_file(trace)!=report['trace_sha256']:
        raise ValueError('Correction replay trace changed')
    m=index['document']['manifest'];bind_media(root,m,verify=True)
    ids=[c['id'] for c in m['cameras']];current=m['initial_program'];cuts=[];count=0
    with trace.open(encoding='utf-8') as source:
        for chunk,arrays in iter_features(root,report['cache_id']):
            for local,t in enumerate(arrays['times']):
                line=source.readline(4*1024*1024+1)
                if not line or len(line)>4*1024*1024:raise ValueError('Missing/oversized correction trace row')
                row=json.loads(line)
                if (abs(row['event_time_s']-float(t))>1e-7 or row.get('source_current_program')!=current
                        or row.get('execute_allowed') is not False):raise ValueError('Correction trace timestamp/PGM mismatch')
                chosen=row['selected_camera'];available=[ids[i] for i in np.flatnonzero(arrays['video_mask'][local]>.5)]
                if (chosen is not None and chosen not in available) or (chosen is None and available):raise ValueError('Correction trace includes invalid selection')
                if chosen is not None and chosen!=current:
                    cuts.append(dict(time_s=float(t),camera_id=chosen));current=chosen
                count+=1
        if source.read(1):raise ValueError('Extra correction trace rows')
    if count!=index['ticks'] or report['variant']!=dict(initial_program=m['initial_program'],cuts=cuts):
        raise ValueError('Correction variant does not match observed trace')
    if report['original_event_id']!=m['original_event_id']:raise ValueError('Correction original event mismatch')
    draft=deepcopy(m);draft['program_events']=cuts;draft['shot_preferences']=[];draft['comparisons']=[]
    draft['trajectory_origin']=dict(kind='offline_policy',checkpoint_sha256=report['checkpoint_sha256'],replay_sha256=digest_file(path),
                                    source_index_sha256=index['index_sha256'],source_content_sha256=index['document']['content_sha256'])
    project=EventStore(root).create(draft)
    return dict(project=project,warning='Unreviewed simulated policy PGM. Re-watch and add expert corrections; old relative labels/A-B/review were NOT copied. Event facts retained only for re-review.',
                production_approved=False,execute_allowed=False)
