"""Feature-space causal hierarchical policy. Not an end-to-end video foundation model."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .contracts_v2 import INTENTS, integer, number, object_fields
from .memory_v2 import STATE_NAMES, append_memory, initial_state
from .model import AttentionBlock

MODEL_FAMILY = 'camix_feature_director_v2'
CHECKPOINT_FORMAT = 'camix_policy_checkpoint_v2'
OBS_NAMES = ('visual', 'audio_rms', 'video_mask', 'audio_mask', 'video_age', 'audio_age',
             'current_program', 'held_seconds', 'committed_cut', 'completed_shot_seconds', 'slow_update', 'style_features')
INPUT_NAMES = OBS_NAMES + STATE_NAMES
OUTPUT_NAMES = ('switch_logit', 'camera_logits', 'intent_logits', 'visibility_logits') + tuple('next_' + n for n in STATE_NAMES)


@dataclass(frozen=True)
class PolicyConfig:
    visual_dim: int = 32
    max_cameras: int = 8
    hidden_dim: int = 64
    heads: int = 4
    layers: int = 2
    short_tokens: int = 16
    shot_tokens: int = 8
    slow_tokens: int = 8
    slow_stride_s: float = 5.
    dropout: float = .1

    def __post_init__(self):
        for name, low, high in [('visual_dim', 1, 4096), ('max_cameras', 1, 16), ('hidden_dim', 8, 512),
                                ('heads', 1, 16), ('layers', 1, 8), ('short_tokens', 1, 128),
                                ('shot_tokens', 1, 64), ('slow_tokens', 1, 64)]:
            integer(getattr(self, name), name, low, high)
        if self.hidden_dim % self.heads:
            raise ValueError('Policy hidden_dim must be divisible by heads')
        number(self.slow_stride_s, 'slow_stride_s', .1, 600)
        number(self.dropout, 'dropout', 0, .5)

    def to_dict(self):
        return asdict(self)


class FeatureDirector(nn.Module):
    def __init__(self, config=PolicyConfig()):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.visual = nn.Sequential(nn.Linear(config.visual_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.fusion = nn.Sequential(nn.Linear(d+7, d), nn.GELU(), nn.LayerNorm(d))
        self.cross_camera = AttentionBlock(d, config.heads, config.dropout)
        self.global_projection = nn.Sequential(nn.Linear(d*2+1, d), nn.GELU())
        self.temporal = nn.ModuleList(AttentionBlock(d, config.heads, config.dropout) for _ in range(config.layers))
        self.memory_types = nn.Parameter(torch.zeros(3, d))
        self.position = nn.Linear(1, d)
        for kind, count in [('short', config.short_tokens), ('shot', config.shot_tokens), ('slow', config.slow_tokens)]:
            self.register_buffer(kind+'_positions', torch.linspace(-1, 0, count).reshape(1, count, 1))
        self.style_projection = nn.Linear(3, d)
        self.intent = nn.Linear(d, len(INTENTS))
        self.intent_embedding = nn.Parameter(torch.randn(len(INTENTS), d)*.02)
        self.visibility = nn.Linear(d, 1)
        self.switch = nn.Sequential(nn.Linear(d*3+1, d), nn.GELU(), nn.Linear(d, 1))
        self.take = nn.Sequential(nn.Linear(d*4+1, d), nn.GELU(), nn.Linear(d, 1))
        self.shot_duration = nn.Linear(1, d)

    def initial_state(self, batch=1, device='cpu'):
        return initial_state(self.config, batch, device)

    def forward(self, visual, audio_rms, video_mask, audio_mask, video_age, audio_age,
                current_program, held_seconds, committed_cut, completed_shot_seconds, slow_update, style_features,
                short_memory, short_mask, shot_memory, shot_mask, slow_memory, slow_mask):
        b, n = video_mask.shape
        v = self.visual(visual) * video_mask[:, :, None]
        hold = torch.log1p(torch.clamp(held_seconds, min=0, max=86400)) / 10
        metadata = torch.stack((torch.log1p(torch.clamp(audio_rms, min=0)*100)*audio_mask,
                                video_mask, audio_mask, torch.clamp(video_age, 0, 30)/30,
                                torch.clamp(audio_age, 0, 30)/30, current_program,
                                hold.expand(-1, n)), -1)
        camera = self.fusion(torch.cat((v, metadata), -1))
        sentinel = torch.zeros_like(camera[:, :1])
        context = torch.cat((camera, sentinel), 1)
        mask = torch.cat((video_mask, torch.ones_like(video_mask[:, :1])), 1)
        camera = self.cross_camera(camera, context, mask) * video_mask[:, :, None]
        pooled = camera.sum(1) / torch.clamp(video_mask.sum(1, keepdim=True), min=1)
        program = (camera*current_program[:, :, None]).sum(1)
        token = self.global_projection(torch.cat((pooled, program, hold), -1))[:, None]
        memories = (short_memory, shot_memory, slow_memory)
        positions = (self.short_positions, self.shot_positions, self.slow_positions)
        past = torch.cat(tuple(m+self.position(p)+self.memory_types[i][None, None] for i, (m, p) in enumerate(zip(memories, positions))), 1)
        history_mask = torch.cat((short_mask, shot_mask, slow_mask), 1)
        for block in self.temporal:
            token = block(token, torch.cat((past, token), 1), torch.cat((history_mask, torch.ones_like(history_mask[:, :1])), 1))
        global_context = token[:, 0] + self.style_projection(style_features)
        intent_logits = self.intent(global_context)
        intent = torch.softmax(intent_logits, -1) @ self.intent_embedding
        visibility = self.visibility(camera).squeeze(-1)
        switch = self.switch(torch.cat((global_context, program, intent, hold), -1))
        take_context = torch.cat((camera, global_context[:, None].expand(-1, n, -1),
                                  program[:, None].expand(-1, n, -1), intent[:, None].expand(-1, n, -1),
                                  torch.sigmoid(visibility)[:, :, None]), -1)
        scores = self.take(take_context).squeeze(-1)
        scores = torch.where(video_mask > .5, scores, torch.full_like(scores, -10000))
        next_short = append_memory(short_memory, short_mask, global_context, torch.ones_like(slow_update))
        # The just-completed shot is summarized from past state, not future labels.
        shot_token = short_memory[:, -1] + self.shot_duration(torch.log1p(torch.clamp(completed_shot_seconds, min=0))/10)
        next_shot = append_memory(shot_memory, shot_mask, shot_token, committed_cut)
        next_slow = append_memory(slow_memory, slow_mask, global_context, slow_update)
        return (switch, scores, intent_logits, visibility) + next_short + next_shot + next_slow


def action_log_probs(switch, camera_logits, video_mask, current_program):
    """KEEP + non-current CUT distribution. Masked all-invalid rows stay finite."""
    current_valid = (video_mask*current_program).sum(1, keepdim=True) > .5
    take_mask = video_mask * (1-current_program) > .5
    first_camera = torch.zeros_like(take_mask); first_camera[:, 0] = True
    safe_take = take_mask | (~take_mask.any(1, keepdim=True) & first_camera)
    target_scores = torch.where(safe_take, camera_logits, torch.full_like(camera_logits, -torch.inf))
    take_log = F.logsigmoid(switch) + F.log_softmax(target_scores, -1)
    keep_log = F.logsigmoid(-switch)
    mask = torch.cat((current_valid, take_mask), 1)
    first_action = torch.zeros_like(mask); first_action[:, 0] = True
    safe_mask = mask | (~mask.any(1, keepdim=True) & first_action)
    logits = torch.where(safe_mask, torch.cat((keep_log, take_log), 1), torch.full_like(torch.cat((keep_log, take_log), 1), -torch.inf))
    probabilities = F.log_softmax(logits, -1)
    return torch.where(mask, probabilities, torch.full_like(probabilities, -10000)), mask
