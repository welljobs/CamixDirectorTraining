from copy import deepcopy
from pathlib import Path
import torch

from camix_director.annotations_v2 import EventStore
from camix_director.config import ModelConfig
from camix_director.contracts_v2 import EVENT_FORMAT, media_identity
from camix_director.model import MemoryDirector
from camix_director.storage import digest_file
from test_studio_data import media_fixture


def make_event(root):
    old = media_fixture(root)
    return dict(format=EVENT_FORMAT, original_event_id='synthetic-v2-original', split='train', source='synthetic',
                duration_s=6., initial_program='a',
                cameras=[dict(id=c['id'], video=c['video'], audio=c['audio'], video_offset_s=0., audio_offset_s=0.,
                              video_delay_s=0., audio_delay_s=0.) for c in old['cameras']],
                clock=dict(mode='strict_live', delay_s=0.),
                license=dict(name='generated unit-test media; no director ground truth', usage_confirmed=True),
                scenario_tags=['synthetic-test'], style=dict(name='unspecified', notes=''),
                program_events=[dict(time_s=2., camera_id='b'), dict(time_s=4., camera_id='a')],
                shot_preferences=[dict(id='shot-1', start_s=1., end_s=5., acceptable_cameras=['b'],
                                       cut_window_s=[2., 3.], context_program=None, intent='unknown', reason_tags=[])],
                events=[dict(id='event-1', start_s=2., end_s=4., label='synthetic marker', visible_cameras=['b'], importance=.5)], comparisons=[])


def reviewed(root, manifest=None):
    store = EventStore(root)
    project = store.create(manifest if manifest is not None else make_event(root))
    return store.review(project['id'], project['revision'], 'synthetic test reviewer (self-declared)', True, True)


def add_comparison(project):
    m = deepcopy(project['manifest'])
    m['comparisons'] = [dict(id='ab-1', start_s=1., end_s=5., source_identity=media_identity(m),
                             a=dict(initial_program='a', cuts=[]),
                             b=dict(initial_program='a', cuts=[dict(time_s=2., camera_id='b')]),
                             label='tie', reviewer='synthetic test reviewer', reason_tags=['rhythm'], notes='fixture')]
    return m


def encoder_weights(root, seed=11):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    config = ModelConfig(max_cameras=2, image_size=16, audio_samples=256, hidden_dim=16,
                         layers=1, memory_tokens=4, dropout=0)
    model = MemoryDirector(config).eval()
    path = Path(root) / 'runs/test-encoder/best.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema_version=1, config=config.to_dict(), model_state=model.state_dict(),
                    qualification='synthetic_smoke_only', production_approved=False), path)
    return path, digest_file(path)
