"""Explicit bounded fast/shot/slow memories. No future events or label tokens."""
import torch

STATE_NAMES = ('short_memory', 'short_mask', 'shot_memory', 'shot_mask', 'slow_memory', 'slow_mask')


def initial_state(config, batch=1, device='cpu'):
    result = []
    for count in (config.short_tokens, config.shot_tokens, config.slow_tokens):
        result.extend((torch.zeros(batch, count, config.hidden_dim, device=device),
                       torch.zeros(batch, count, device=device)))
    return tuple(result)


def append_memory(memory, mask, token, gate):
    """Traceable tensor gate: shot/slow state changes only on committed signals."""
    candidate = torch.cat((memory[:, 1:], token[:, None]), 1)
    candidate_mask = torch.cat((mask[:, 1:], torch.ones_like(mask[:, :1])), 1)
    g = gate.reshape(-1, 1, 1)
    return memory * (1-g) + candidate * g, mask * (1-g[:, :, 0]) + candidate_mask * g[:, :, 0]


def detach_state(state):
    return tuple(t.detach() for t in state)
