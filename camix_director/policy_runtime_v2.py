"""Causal tensor preparation and suggestion-only streaming. No mixer/network control."""
from dataclasses import dataclass
import math
import threading
import numpy as np
import torch

from .contracts_v2 import fingerprint, number, object_fields, text
from .model_v2 import OBS_NAMES, action_log_probs
from .policy_io_v2 import load_policy


@dataclass
class CausalSignals:
    last_program: int | None = None
    last_held: float = 0.
    last_time: float | None = None
    last_slow: float | None = None

    def values(self, time_s, program, held, stride):
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError('Non-increasing policy input timestamp')
        cut = self.last_program is not None and program is not None and self.last_program != program
        duration = max(0., self.last_held + (time_s-self.last_time) - held) if cut else 0.
        slow = self.last_slow is None or time_s-self.last_slow >= stride-1e-8
        if slow:
            self.last_slow = time_s
        self.last_program, self.last_held, self.last_time = program, held, time_s
        return float(cut), duration, float(slow)


def tensors_for_step(observations, position, config, device, signals, clock, ablation='full'):
    """Build from observed media/PGM only. Targets/events/intent labels are not read."""
    t = float(observations['times'][position])
    cutoff = t + clock['delay_s']
    n = observations['video_mask'].shape[-1]
    if n > config.max_cameras or observations['visual'].shape[-1] != config.visual_dim:
        raise ValueError('Feature-space/camera count does not match policy')
    values = {}
    for key in ('visual', 'audio_rms', 'video_mask', 'audio_mask', 'current_program'):
        array = np.asarray(observations[key][position], np.float32)
        shape = (config.max_cameras, config.visual_dim) if key == 'visual' else (config.max_cameras,)
        padded = np.zeros(shape, np.float32);padded[:n] = array
        values[key] = torch.from_numpy(padded[None]).to(device)
    for kind in ('video', 'audio'):
        end = observations[kind+'_end_s'][position]
        age = np.where(observations[kind+'_mask'][position] > .5, cutoff-end, 30.)
        padded = np.full(config.max_cameras, 30., np.float32);padded[:n] = age
        values[kind+'_age'] = torch.from_numpy(padded[None]).to(device)
    held = float(observations['held_seconds'][position, 0])
    pgm = np.flatnonzero(observations['current_program'][position] > .5)
    cut, duration, slow = signals.values(t, int(pgm[0]) if len(pgm) else None, held, config.slow_stride_s)
    for key, value in [('held_seconds', held), ('committed_cut', cut), ('completed_shot_seconds', duration), ('slow_update', slow)]:
        values[key] = torch.tensor([[value]], dtype=torch.float32, device=device)
    style = np.asarray(observations.get('style_features', np.zeros((len(observations['times']),3),np.float32))[position],np.float32)
    values['style_features'] = torch.as_tensor(style[None],device=device)
    if ablation == 'video_content_muted':
        values['visual'] = torch.zeros_like(values['visual'])
    elif ablation == 'audio_muted':
        values['audio_rms'] = torch.zeros_like(values['audio_rms'])
        values['audio_mask'] = torch.zeros_like(values['audio_mask'])
    elif ablation not in ('full', 'memory_reset_each_step'):
        raise ValueError('Unknown policy ablation')
    return tuple(values[name] for name in OBS_NAMES), values


def tensor_targets(targets, position, config, device):
    result = {}
    camera_fields = {'acceptable_cameras', 'policy_acceptable_cameras', 'visibility_target', 'visibility_mask'}
    for key, value in targets.items():
        array = np.asarray(value[position])
        if key in camera_fields:
            padded = np.zeros(config.max_cameras, np.float32);padded[:len(array)] = array;array = padded
        result[key] = torch.as_tensor(np.array(array, copy=True), device=device).unsqueeze(0)
    return result


class StreamingDirector:
    """Explicit caller-provided actual PGM. Suggestions never execute TAKE."""
    def __init__(self, checkpoint, min_hold_s=0., min_probability=0., max_gap_s=30.):
        self.model, self.checkpoint = load_policy(checkpoint)
        self.min_hold = number(min_hold_s, 'minimum hold', 0, 60)
        self.min_probability = number(min_probability, 'experimental probability threshold', 0, 1)
        self.max_gap = number(max_gap_s, 'reset gap', .1, 600)
        self.lock = threading.Lock()
        self.reset()

    def _reset_unlocked(self):
        self.memory = self.model.initial_state()
        self.signals = CausalSignals()
        self.generation = None
        self.camera_ids = None
        self.style_features = None

    def reset(self):
        with self.lock:
            self._reset_unlocked()

    @torch.no_grad()
    def step(self, packet):
        with self.lock:
            fields = ('format', 'event_time_s', 'clock', 'generation', 'feature_space_id', 'camera_ids', 'current_program',
                      'held_seconds', 'visual', 'audio_rms', 'video_mask', 'audio_mask', 'video_end_s', 'video_available_s',
                      'audio_end_s', 'audio_available_s', 'style_features')
            object_fields(packet, fields, fields)
            if packet['format'] != 'camix_feature_packet_v2' or packet['feature_space_id'] != self.checkpoint['feature_space_id']:
                raise ValueError('Shadow packet feature-space identity mismatch')
            clock = packet['clock']
            object_fields(clock, ('mode', 'delay_s'), ('mode', 'delay_s'))
            delay = number(clock['delay_s'], 'clock delay', 0, 30)
            if clock['mode'] not in ('strict_live', 'fixed_delay') or (clock['mode'] == 'strict_live' and delay):
                raise ValueError('Invalid shadow clock')
            if clock not in self.checkpoint['clock_profiles']:
                raise ValueError('Clock profile was not validated with this checkpoint')
            t = number(packet['event_time_s'], 'event time')
            generation = text(packet['generation'], 'generation', 160)
            ids = packet['camera_ids']
            if (not isinstance(ids, list) or not ids or len(ids) > self.model.config.max_cameras
                    or any(not isinstance(c, str) or not c or len(c) > 100 for c in ids) or len(set(ids)) != len(ids)):
                raise ValueError('Invalid camera mapping')
            if packet['current_program'] is not None and packet['current_program'] not in ids:
                raise ValueError('Unknown actual program camera')
            held = number(packet['held_seconds'], 'held time')
            n = len(ids)
            style = np.asarray(packet['style_features'],np.float32)
            if style.shape!=(3,) or not np.isfinite(style).all() or np.any(np.abs(style)>1):
                raise ValueError('Style condition must be three finite axes in [-1,1]')
            arrays = {}
            for name in fields[8:]:
                if name in ('held_seconds','style_features'):
                    continue
                expected = (n, self.model.config.visual_dim) if name == 'visual' else (n,)
                value = np.asarray(packet[name], np.float32 if name in ('visual', 'audio_rms', 'video_mask', 'audio_mask') else np.float64)
                if value.shape != expected or not np.isfinite(value).all():
                    raise ValueError('Invalid finite shadow array: '+name)
                arrays[name] = value[None]
            for kind in ('video', 'audio'):
                mask = arrays[kind+'_mask']
                if not np.isin(mask, [0, 1]).all():
                    raise ValueError('Binary modality mask required')
                valid = mask > .5
                end, available = arrays[kind+'_end_s'], arrays[kind+'_available_s']
                if np.any(valid & ((end > t+delay+1e-6) | (available > t+delay+1e-6) | (available < end-1e-6))):
                    raise ValueError('Future/unpublished shadow input')
            if np.any(arrays['visual'][arrays['video_mask'] < .5] != 0) or np.any(arrays['audio_rms'][arrays['audio_mask'] < .5] != 0):
                raise ValueError('Invalid modality must have zero features')
            if np.any(arrays['audio_rms'] < 0) or np.any(arrays['audio_rms'] > 1.0001):
                raise ValueError('Invalid audio energy')
            reset_reason = None
            if generation != self.generation:
                self._reset_unlocked();self.generation = generation;self.camera_ids = list(ids);self.style_features=style.tolist();reset_reason = 'generation'
            elif self.style_features != style.tolist():
                raise ValueError('Style condition changed without new generation')
            elif self.camera_ids != ids:
                raise ValueError('Camera slots changed without a new generation')
            elif self.signals.last_time is not None and t <= self.signals.last_time:
                raise ValueError('Duplicate/backward timestamp; reset generation on seek')
            elif self.signals.last_time is not None and t-self.signals.last_time > self.max_gap:
                self.memory = self.model.initial_state();self.signals = CausalSignals();reset_reason = 'gap'
            current = np.zeros((1, n), np.float32)
            if packet['current_program'] is not None:
                current[0, ids.index(packet['current_program'])] = 1
            observations = dict(arrays, times=np.array([t]), current_program=current, held_seconds=np.array([[held]]),style_features=style[None])
            inputs, values = tensors_for_step(observations, 0, self.model.config, 'cpu', self.signals, clock)
            result = self.model(*inputs, *self.memory)
            if not all(torch.isfinite(v).all() for v in result):
                raise ValueError('Non-finite policy result')
            self.memory = result[4:]
            logp, legal = action_log_probs(result[0], result[1], values['video_mask'], values['current_program'])
            available = np.flatnonzero(arrays['video_mask'][0] > .5).tolist()
            raw, selected, reason = None, None, 'manual_intervention_no_video'
            probability = None
            if available:
                choice = int(logp.masked_fill(~legal, -torch.inf).argmax(1)[0])
                raw = packet['current_program'] if choice == 0 else ids[choice-1]
                probability = float(torch.exp(logp[0, choice]))
                selected, reason = raw, 'learned_suggestion'
                current_valid = packet['current_program'] in [ids[s] for s in available]
                if not current_valid:
                    selected = raw if raw in [ids[s] for s in available] else ids[available[0]]
                    reason = 'signal_fallback'
                elif raw != packet['current_program'] and (held < self.min_hold or probability < self.min_probability):
                    selected = packet['current_program'];reason = 'experimental_guard'
            return dict(event_time_s=t, decision_time_s=t+delay, raw_candidate_camera=raw, selected_camera=selected,
                        source_current_program=packet['current_program'], reason=reason, reset_reason=reset_reason,
                        raw_action_probability=probability, probability_calibrated=False,
                        intent_distribution=torch.softmax(result[2], -1)[0].tolist(),
                        interpretation='intent prediction, not a verified causal explanation',
                        production_approved=False, execute_allowed=False)
