from __future__ import annotations
from transformers import AutoConfig, AutoModel
from .ecg import ECGConfig, ECGModel

def build_model(cfg):
    name = cfg.model.name

    if name == "mnist_classifier":
        from .mnist import MNISTClassifier
        return MNISTClassifier(cfg)

    elif name == "ecg" or name == "lead_model":
        model_cfg = ECGConfig(
            task=cfg.model.task,
            hidden_dim=cfg.model.hidden_dim,
            conv_width=cfg.model.conv_width,
            dropout=cfg.model.dropout,
            lead=getattr(cfg.model, "lead", None),
        )
        model = ECGModel(model_cfg)
        return model

    else:
        raise ValueError(f"Model {name} not recognized.")
