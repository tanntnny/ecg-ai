from __future__ import annotations

from pathlib import Path
from copy import copy

from ..core.logger import logger

from .hf_trainer import HFTrainer
from ..models.ecg import CNNLeadEncoder, ECGModelHF
from ..interfaces.protocol import TrainerProtocol

_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class LeadTrainer(TrainerProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = Path(self.cfg.output_dir)
    
    def fit(self):
        for lead in _LEAD_ORDER:
            lead_output_dir = self.output_dir / lead
            lead_output_dir.mkdir(parents=True, exist_ok=True)

            logger.log_info("LeadTrainer", f"Starting training for lead {lead}")
            lead_cfg = copy(self.cfg)
            lead_cfg.model.lead = lead
            lead_cfg.output_dir = str(lead_output_dir)

            hf_trainer = HFTrainer(lead_cfg)
            hf_trainer.fit()

class ECGLeadFinetuner(TrainerProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = Path(self.cfg.output_dir)
    
    def fit(self):
        logger.log_info("ECGLeadFinetuner", "Starting fine-tuning")
        for lead in _LEAD_ORDER:
            
            lead_model = LeadTrainer()
            logger.log_info("ECGLeadFinetuner", f"Starting fine-tuning")
            hf_trainer = HFTrainer(self.cfg)
            hf_trainer.fit()