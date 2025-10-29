from __future__ import annotations

# ---------------- Build model ----------------
def build_model(cfg):
    if cfg.model.name == "mnist_classifier":
        from .mnist import MNISTClassifier
        return MNISTClassifier(cfg)
    elif cfg.model.name == "ecg":
        from .ecg import ECGModel
        return ECGModel(cfg)
    elif cfg.model.name == "lead_model":
        from .ecg import LeadModel
        return LeadModel(cfg)
    else:
        raise ValueError(f"Model {cfg.model.name} not recognized.")