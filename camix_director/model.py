"""One causal step: raw RGB + mono PCM -> preference values + bounded memory.

No scene classes, no camera-identity embeddings, no KEEP/CUT classification head.
The memory contains learned previous-step summaries, not an unbounded KV cache.
"""
import math
import torch
from torch import nn
from .config import ModelConfig


class TinyVideoEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, width))

    def forward(self, x):
        return self.net(x)


class MobileVideoEncoder(nn.Module):
    def __init__(self, width, pretrained):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(576, width)
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer("std",torch.tensor([.229,.224,.225]).reshape(1,3,1,1))

    def forward(self, x):
        return self.projection(self.pool(self.features(((x+1)*.5-self.mean)/self.std)).flatten(1))


class AudioEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 31, stride=8, padding=15), nn.ReLU(),
            nn.Conv1d(8, 16, 15, stride=8, padding=7), nn.ReLU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, width))

    def forward(self, x):
        return self.net(x)


class NormalizedAudioEncoder(nn.Module):
    """Gain-robust waveform features plus learned log-energy features, not a rule gate."""
    def __init__(self,width):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv1d(1,8,31,stride=8,padding=15),nn.ReLU(),
            nn.Conv1d(8,16,15,stride=8,padding=7),nn.ReLU(),
            nn.Conv1d(16,32,7,stride=4,padding=3),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.projection=nn.Linear(34,width)
    def forward(self,x):
        rms=torch.sqrt(torch.mean(x*x,dim=-1)+1e-10)
        peak=torch.amax(torch.abs(x),dim=-1)
        normalized=torch.clamp(x/(rms.unsqueeze(-1)+1e-5),-10,10)
        energy=torch.cat((torch.log(rms)/10,torch.log(peak+1e-5)/10),dim=-1)
        return self.projection(torch.cat((self.features(normalized),energy),dim=-1))


class AttentionBlock(nn.Module):
    """Primitive tensor attention deliberately avoids CUDA-only fused kernels."""
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_mask):
        b, qlen, width = query.shape
        klen = context.shape[1]
        q = self.q(self.query_norm(query)).reshape(b, qlen, self.heads, self.head_width).transpose(1, 2)
        normalized = self.context_norm(context)
        k = self.k(normalized).reshape(b, klen, self.heads, self.head_width).transpose(1, 2)
        v = self.v(normalized).reshape(b, klen, self.heads, self.head_width).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        logits = logits + (1.0 - context_mask[:, None, None, :]) * -10000.0
        weights = torch.softmax(logits, dim=-1)
        result = torch.matmul(weights, v).transpose(1, 2).reshape(b, qlen, width)
        x = query + self.dropout(self.out(result))
        return x + self.dropout(self.ff(self.ff_norm(x)))


class MemoryDirector(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.video = (MobileVideoEncoder(d, config.pretrained_visual)
                      if config.visual_encoder == "mobilenet_v3_small" else TinyVideoEncoder(d))
        self.audio = (NormalizedAudioEncoder(d) if config.audio_encoder=="normalized_cnn" else AudioEncoder(d))
        self.fusion = nn.Sequential(nn.Linear(d * 2 + 6, d), nn.GELU(), nn.LayerNorm(d))
        self.cross_camera = AttentionBlock(d, config.heads, config.dropout)
        self.global_projection = nn.Sequential(nn.Linear(d * 2 + 1, d), nn.GELU())
        self.time_projection = nn.Linear(1, d)
        self.temporal = nn.ModuleList(AttentionBlock(d, config.heads, config.dropout) for _ in range(config.layers))
        self.keep_head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
        self.take_head = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("memory_positions", torch.linspace(-1.0, 0.0, config.memory_tokens).reshape(1, -1, 1))

    def initial_memory(self, batch=1, device="cpu"):
        return (torch.zeros(batch, self.config.memory_tokens, self.config.hidden_dim, device=device),
                torch.zeros(batch, self.config.memory_tokens, device=device))

    def forward(self, video_frames, audio_pcm, modality_mask, camera_mask, media_age,
                current_program, held_seconds, memory, memory_mask):
        b, n = camera_mask.shape
        d = self.config.hidden_dim
        visual = self.video(video_frames.reshape(b * n, 3, self.config.image_size, self.config.image_size)).reshape(b, n, d)
        audio = self.audio(audio_pcm.reshape(b * n, 1, self.config.audio_samples)).reshape(b, n, d)
        visual = visual * modality_mask[:, :, 0:1]
        audio = audio * modality_mask[:, :, 1:2]
        hold = torch.log1p(torch.clamp(held_seconds, min=0.0)) / 5.0
        metadata = torch.cat((modality_mask, torch.clamp(media_age, 0, 10) / 10,
                              current_program.unsqueeze(-1), hold.unsqueeze(1).expand(-1, n, -1)), dim=-1)
        camera = self.fusion(torch.cat((visual, audio, metadata), dim=-1))
        # A valid zero sentinel prevents NaNs if every camera is unavailable.
        sentinel = torch.zeros_like(camera[:, :1, :])
        context = torch.cat((camera, sentinel), dim=1)
        context_mask = torch.cat((camera_mask, torch.ones_like(camera_mask[:, :1])), dim=1)
        camera = self.cross_camera(camera, context, context_mask) * camera_mask.unsqueeze(-1)
        pooled = camera.sum(dim=1) / torch.clamp(camera_mask.sum(dim=1, keepdim=True), min=1)
        program = (camera * current_program.unsqueeze(-1)).sum(dim=1)
        token = self.global_projection(torch.cat((pooled, program, hold), dim=-1)).unsqueeze(1)
        # Past summaries only: no future frame is ever available to this forward step.
        history = memory + self.time_projection(self.memory_positions)
        for block in self.temporal:
            context = torch.cat((history, token), dim=1)
            mask = torch.cat((memory_mask, torch.ones_like(memory_mask[:, :1])), dim=1)
            token = block(token, context, mask)
        global_context = token[:, 0, :]
        keep = self.keep_head(torch.cat((program, global_context), dim=-1))
        take_context = torch.cat((camera, global_context[:, None, :].expand(-1, n, -1),
                                  program[:, None, :].expand(-1, n, -1)), dim=-1)
        take = self.take_head(take_context).squeeze(-1)
        take = torch.where(current_program > 0.5, keep.expand(-1, n), take)
        take = torch.where(camera_mask > 0.5, take, torch.full_like(take, -10000.0))
        gains = take - keep
        next_memory = torch.cat((memory[:, 1:, :], token), dim=1)
        next_mask = torch.cat((memory_mask[:, 1:], torch.ones_like(memory_mask[:, :1])), dim=1)
        return keep, take, gains, next_memory, next_mask
