from __future__ import annotations

def build_pipeline(cfg):
    if cfg.pipeline.pipeline == "preprocess_ecg":
        from ..pipeline.preprocess_ecg import PreprocessECGPipeline
        return PreprocessECGPipeline(cfg)
    else:
        raise ValueError(f"Unknown pipeline type: {cfg.pipeline.pipeline}")