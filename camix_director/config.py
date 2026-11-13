from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1
INPUT_NAMES = ("video_frames", "audio_pcm", "modality_mask", "camera_mask", "media_age", "current_program", "held_seconds", "memory", "memory_mask")
OUTPUT_NAMES = ("keep_value", "take_value", "switch_gain", "next_memory", "next_memory_mask")

@dataclass(frozen=True)
class ModelConfig:
    max_cameras: int = 4
    image_size: int = 96
    sample_rate: int = 16000
    audio_samples: int = 16000
    decision_hz: float = 5.0
    hidden_dim: int = 64
    heads: int = 4
    layers: int = 2
    memory_tokens: int = 32
    dropout: float = 0.1
    visual_encoder: str = "tiny_cnn"
    pretrained_visual: bool = False
    audio_encoder: str = "raw_cnn"

    def __post_init__(self):
        if self.max_cameras < 1 or self.image_size < 16 or self.audio_samples < 256:
            raise ValueError("invalid media dimensions")
        if self.hidden_dim < 8 or self.heads < 1 or self.hidden_dim % self.heads or self.layers < 1 or self.memory_tokens < 1:
            raise ValueError("invalid Transformer/memory dimensions")
        if self.sample_rate < 1 or self.decision_hz <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid rates/dropout")
        if self.audio_encoder not in ("raw_cnn","normalized_cnn"):
            raise ValueError("unsupported audio encoder")
        if self.visual_encoder not in ("tiny_cnn", "mobilenet_v3_small"):
            raise ValueError("unsupported visual encoder")

    def to_dict(self):
        return asdict(self)
