from __future__ import annotations

from typing import Optional, Dict, Tuple, List, Any
from pathlib import Path

import os
import numpy as np
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset

from ..interfaces.protocol import DataProtocol
from ..core.logger import logger

_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
_SEX_MAP = {'M': 0, 'F': 1}

def _get_basename(file_path: str | Path) -> str:
    split = os.path.basename(file_path).split('_')
    return split[0] + '_' + split[1]

class ECGDataset(Dataset):
    def __init__(self, data_config: str | Path, basenames: List[str], task: str):
        self.data_config = pd.read_csv(data_config)
        self.sample: List[Dict[str, Any]] = [] # List[Dict[str, [T, F]]]
        
        group_by_name: Dict[str, List] = {}
        for _, row in self.data_config.iterrows():
            lead, src = row["lead"], row["src"]
            for _, s in pd.read_csv(src).iterrows():
                waveform, logmel, age, sex = s["waveform"], s["logmel"], s["age"], s["sex"]
                cv_risk = s.get("cv_risk", None)

                if pd.isna(waveform) or pd.isna(logmel) or pd.isna(age) or pd.isna(sex) or pd.isna(cv_risk):
                    continue

                sex = _SEX_MAP[sex]
                cv_risk = float(cv_risk)
                logmel_tensor = torch.load(logmel) # [T, F]
                # waveform_tensor, _ = torchaudio.load(waveform) # [T, ]
                waveform_tensor = torch.zeros_like(logmel_tensor)
                basename = _get_basename(logmel)

                if basename not in basenames:
                    continue
                
                if basename not in group_by_name:
                    group_by_name[basename] = [{}, {}, age, sex, cv_risk]
                group_by_name[basename][0][lead] = logmel_tensor
                group_by_name[basename][1][lead] = waveform_tensor
        
        for name, [leads_logmel, leads_waveform, age, sex, cv_risk] in group_by_name.items():
            if set(leads_logmel.keys()) != set(leads_waveform.keys()) != set(_LEAD_ORDER):
                continue
            self.sample.append({
                "logmel": leads_logmel,
                "waveform": leads_waveform,
                "age_label": torch.tensor(age, dtype=torch.long),
                "sex_label": torch.tensor(sex, dtype=torch.long),
                "label": torch.tensor(age if task == "age" else sex, dtype=torch.long),
                "meta": {
                    "basename": name,
                    "cv_risk": torch.tensor(cv_risk, dtype=torch.float)
                }
            })

    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sample[idx]

def collator(batch: List[Dict[str, Any]]):
    # x: Dict[lead, tensor]
    waveforms: dict = {lead: [] for lead in _LEAD_ORDER} # lead: [B, T, F]
    logmels: dict = {lead: [] for lead in _LEAD_ORDER} # lead: [B, T, F]
    age_labels = [] # [B, ]
    sex_labels = [] # [B, ]
    labels = [] # [B, ]
    for sample in batch:
        logmel, waveform, age, sex, label = sample["logmel"], sample["waveform"], sample["age_label"], sample["sex_label"], sample["label"]
        for lead in _LEAD_ORDER:
            waveforms[lead].append(waveform[lead].unsqueeze(0)) # [1, T, F]
            logmels[lead].append(logmel[lead].unsqueeze(0)) # [1, T, F]
        age_labels.append(age.unsqueeze(0)) # [1, ]
        sex_labels.append(sex.unsqueeze(0)) # [1, ]
        labels.append(label.unsqueeze(0)) # [1, ]
    for lead in _LEAD_ORDER:
        waveforms[lead] = torch.cat(waveforms[lead], dim=0) # [B, T, F]
        logmels[lead] = torch.cat(logmels[lead], dim=0) # [B, T, F]
    age_labels = torch.cat(age_labels, dim=0) # [B, ]
    sex_labels = torch.cat(sex_labels, dim=0) # [B, ]
    labels = torch.cat(labels, dim=0) # [B, ]
    return {
        "waveforms": waveforms,
        "logmels": logmels,
        "age_labels": age_labels,
        "sex_labels": sex_labels,
        "labels": labels
    }

class ECGData(DataProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logger
        
        self.data_confg = pd.read_csv(self.cfg.data.src)
        self.basenames = self._get_all_basenames()
        self.train_ratio = self.cfg.data.train_ratio
        train_size = int(len(self.basenames) * self.train_ratio)
        
        self.logger.log_info("ECGData", f"Total samples: {len(self.basenames)}, Train size: {train_size}, Eval size: {len(self.basenames) - train_size}")
        
        self.train_dataset = ECGDataset(
            data_config=self.cfg.data.src,
            basenames=self.basenames[:train_size],
            task=self.cfg.model.task,
        )
        self.eval_dataset = ECGDataset(
            data_config=self.cfg.data.src,
            basenames=self.basenames[train_size:],
            task=self.cfg.model.task,
        )

    def _get_all_basenames(self) -> List[str]:
        basenames = set()
        for _, row in self.data_confg.iterrows():
            src = row["src"]
            for _, s in pd.read_csv(src).iterrows():
                basename = _get_basename(s["logmel"])
                basenames.add(basename)
        return list(basenames)

    def get_collator(self):
        return collator
    
    def get_train_dataset(self) -> Dataset:
        return self.train_dataset
    
    def get_eval_dataset(self) -> Dataset:
        return self.eval_dataset

