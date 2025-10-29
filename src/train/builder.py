from __future__ import annotations

from ..interfaces.protocol import TrainerProtocol
from .hf_trainer import HFTrainer

# ---------------- Build trainer ----------------
def build_trainer(cfg) -> TrainerProtocol:
    trainer = cfg.train.trainer
    if trainer == "hf":
        return HFTrainer(cfg)
    if trainer == "lead_encoder":
        return LeadEncoderTrainer(cfg)
    else:
        raise ValueError(f"Unknown trainer type: {trainer}")