"""Set-valued and masked supervision. Uncertain/unannotated never becomes KEEP."""
import torch
from torch.nn import functional as F
from .model_v2 import action_log_probs

DEFAULT_WEIGHTS = dict(action=1., switch=.2, intent=.1, visibility=.1)


def policy_losses(outputs, observations, targets, weights=None):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    switch, camera, intent, visibility = outputs[:4]
    logp, legal = action_log_probs(switch, camera, observations['video_mask'], observations['current_program'])
    accepted = targets.get('policy_acceptable_cameras', targets['acceptable_cameras']) > .5
    current = observations['current_program'] > .5
    good = torch.cat(((accepted & current).any(1, keepdim=True), accepted & ~current), 1) & legal
    rows = targets.get('action_annotated', targets['annotated']).bool() & good.any(1)
    masked = torch.where(good, logp, torch.full_like(logp, -10000))
    action_values = -torch.logsumexp(masked, 1)
    loss = switch.sum()*0
    metrics = {}
    def add(name, values, mask):
        nonlocal loss
        count = float(mask.sum().detach())
        if count:
            total = (values*mask).sum()
            loss = loss + weights[name]*total/count
            metrics[name] = dict(sum=float(total.detach()), count=count)
    add('action', action_values, rows)
    binary = F.binary_cross_entropy_with_logits(switch[:, 0], targets['switch_target'], reduction='none')
    add('switch', binary, targets['switch_mask'].bool())
    intent_loss = -(targets['intent_distribution']*F.log_softmax(intent, -1)).sum(1)
    add('intent', intent_loss, targets['intent_mask'].bool())
    visibility_loss = F.binary_cross_entropy_with_logits(visibility, targets['visibility_target'], reduction='none')
    add('visibility', visibility_loss, targets['visibility_mask'])
    choice = logp.argmax(1)
    metrics['correct'] = int((good.gather(1, choice[:, None])[:, 0] & rows).sum().detach())
    metrics['action_count'] = int(rows.sum().detach())
    return loss if any(k in metrics for k in DEFAULT_WEIGHTS) else None, metrics


def distillation_loss(student, teacher, observations, temperature=2.):
    sp, mask = action_log_probs(student[0]/temperature, student[1]/temperature,
                                observations['video_mask'], observations['current_program'])
    tp, _ = action_log_probs(teacher[0]/temperature, teacher[1]/temperature,
                            observations['video_mask'], observations['current_program'])
    valid = mask.any(1)
    if not valid.any():
        return student[0].sum()*0
    terms = torch.exp(tp)*(tp-sp)
    return (terms*mask).sum(1)[valid].mean()*temperature**2
