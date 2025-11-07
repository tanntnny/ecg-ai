from __future__ import annotations

from typing import Optional, Dict, Tuple, List, Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..core.logger import logger

# ---------------- Constant ----------------
_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class CNNLeadEncoder(nn.Module):
    """
    Expected input shape: [B, T, F]
    """
    def __init__(
        self,
        in_ch: int = 1,
        hid_dim: int = 512,
        conv_widths: Optional[list] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hid_dim
        chs = [in_ch, 32, 64, 128, 256]
        k = 5
        p = k // 2

        self.block1 = nn.Sequential(
            nn.Conv2d(chs[0], chs[1], kernel_size=(k, k), padding=(p, p)),
            nn.BatchNorm2d(chs[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(chs[1], chs[2], kernel_size=(k, k), padding=(p, p)),
            nn.BatchNorm2d(chs[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(chs[2], chs[3], kernel_size=(k, k), padding=(p, p)),
            nn.BatchNorm2d(chs[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(chs[3], chs[4], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(chs[4]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[4], 4 * hid_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(4 * hid_dim),
            nn.Linear(4 * hid_dim, hid_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.block1(x)  # [B, C1, T/2, F/2]
        x = self.block2(x)  # [B, C2, T/4, F/4]
        x = self.block3(x)  # [B, C3, T/8, F/8]
        x = self.block4(x)  # [B, C4, T/8, F/8]

        if mask is not None:
            if mask.dim() == 3:
                mask_t = (mask > 0).any(dim=-1).float()
            elif mask.dim() == 2:
                mask_t = mask.float()
            else:
                raise ValueError(f"mask must be [B, T] or [B, T, F], got {tuple(mask.shape)}")
            m = mask_t.unsqueeze(1).unsqueeze(-1)
            m = F.interpolate(m, size=(x.shape[2], 1), mode="nearest")
            x = x * (m > 0.5).to(x.dtype)

        x = self.gap(x)
        x = self.proj(x)
        return x
    
class ECGEncoder(nn.Module):
    def __init__(self, cfg):
        super(ECGEncoder, self).__init__()
        self.cfg = cfg

        self.cnn_branch = nn.ModuleDict({
            lead: CNNLeadEncoder(
                in_ch=1,
                hid_dim=self.cfg.model.hidden_dim,
                conv_widths=self.cfg.model.conv_width,
                dropout=self.cfg.model.dropout,
            )
            for lead in _LEAD_ORDER
        })
        
    def forward(
        self,
        logmels: Dict[str, torch.Tensor], # Dict[lead, logmels]
    ):
        # x: Dict[lead, tensor]
        out = []
        for lead, tensor in logmels.items():
            out.append(self.cnn_branch[lead](tensor))
        out = torch.cat(out, dim=-1)
        return out
    
    def get_hidden_dim(self) -> int:
        return self.cfg.model.hidden_dim * 12
    
    def freeze_cnn_branches(self, lead: str) -> None:
        for param in self.cnn_branch[lead].parameters():
            param.requires_grad = False

    def save_lead_encoder(self, lead: str, path: str) -> None:
        torch.save(self.cnn_branch[lead].state_dict(), path)
    
    def load_lead_encoder(self, lead: str, path: str) -> None:
        self.cnn_branch[lead].load_state_dict(torch.load(path))

class AgeHead(nn.Module):
    def __init__(self, cfg, input_dim: int = 1024):
        super(AgeHead, self).__init__()
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(512, 1)
        )
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        loss = F.mse_loss(logits.squeeze(-1), labels.float())
        return {
            "logits": logits,
            "loss": loss,
        }

class SexHead(nn.Module):
    def __init__(self, cfg, input_dim: int = 1024):
        super(SexHead, self).__init__()
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.cfg.model.dropout),
            nn.Linear(512, 2)
        )
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        loss = F.cross_entropy(logits.squeeze(-1), labels.long())
        return {
            "logits": logits,
            "loss": loss,
        }

class LeadModel(nn.Module):
    def __init__(self, cfg):
        super(LeadModel, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.lead = self.cfg.model.lead
        self.lead_encoder = CNNLeadEncoder(
            in_ch=1,
            hid_dim=self.cfg.model.hidden_dim,
            conv_widths=self.cfg.model.conv_width,
            dropout=self.cfg.model.dropout,
        )
        
        if self.cfg.model.task == "age":
            self.head = AgeHead(cfg, input_dim=self.lead_encoder.hidden_dim)
        elif self.cfg.model.task == "sex":
            self.head = SexHead(cfg, input_dim=self.lead_encoder.hidden_dim)
        else:
            raise ValueError(f"Unknown task: {self.cfg.model.task}")

        n_params = sum([p.numel() for p in self.parameters()])
        trainable_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        self.logger.log_info("LeadModel", f"Initialized ECGModel with task={self.cfg.model.task} for lead={self.lead}")
        self.logger.log_info("LeadModel", f"Total parameters: {n_params/1e6:.2f}M, Trainable parameters: {trainable_params/1e6:.2f}M")
    
    def forward(
        self,
        logmels: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        **kwargs,
    ):
        enc_out = self.lead_encoder(logmels[self.lead])
        head_out = self.head(x=enc_out, labels=labels)
        return head_out
    
    def save_pretrained(self, path: str) -> None:
        self.logger("LeadModel", f"Saving lead encoder for lead {self.lead} to {path}...")
        torch.save(self.lead_encoder.state_dict(), path)
        

class ECGModel(nn.Module):
    def __init__(self, cfg):
        super(ECGModel, self).__init__()
        
        self.cfg = cfg
        self.encoder = ECGEncoder(cfg)
        self.logger = logger
        
        if self.cfg.model.task == "age":
            self.head = AgeHead(cfg, input_dim=self.encoder.get_hidden_dim())
        elif self.cfg.model.task == "sex":
            self.head = SexHead(cfg, input_dim=self.encoder.get_hidden_dim())
        else:
            raise ValueError(f"Unknown task: {self.cfg.model.task}")
        
        n_params = sum([p.numel() for p in self.parameters()])
        trainable_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        self.logger.log_info("ECGModel", f"Initialized ECGModel with task={self.cfg.model.task}")
        self.logger.log_info("ECGModel", f"Total parameters: {n_params/1e6:.2f}M, Trainable parameters: {trainable_params/1e6:.2f}M")
    
    def forward(
        self,
        waveforms: Dict[str, torch.Tensor],
        logmels: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        **kwargs,
    ):
        enc_out = self.encoder(logmels=logmels)
        head_out = self.head(x=enc_out, labels=labels)
        return head_out
    
    def save_model(self, path: str) -> None:
        self.logger("ECGModel", f"Saving model to {path}...")
        torch.save(self.state_dict(), path)