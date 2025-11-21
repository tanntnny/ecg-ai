from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig, OmegaConf

import warnings

from .train.builder import build_trainer
from .inference.builder import build_inferencer
from .pipeline.builder import build_pipeline

@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())
    cmd = cfg.get("cmd", None)
    
    if cfg.get("debug", False):
        warnings.filterwarnings("ignore", category=UserWarning)
    
    if cmd == "train":
        trainer = build_trainer(cfg)
        trainer.fit()
    elif cmd == "inference":
        inferencer = build_inferencer(cfg)
        inferencer.inference()
    elif cmd == "pipeline":
        pipeline = build_pipeline(cfg)
        pipeline.run()
    else:
        raise SystemExit(f"Unknown cmd={cmd}")

    save_user_config(cfg, cfg.output_dir)

def save_user_config(cfg: DictConfig, save_dir: str) -> None:
    save_path = os.path.join(save_dir, "user_configs.yaml")
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Config saved to {save_path}")

if __name__ == "__main__":
    main()