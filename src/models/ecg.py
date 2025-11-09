from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

# ---------------- Constants ----------------
_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ---------------- Config ----------------
class ECGConfig(PretrainedConfig):
    model_type = "ecg_model"

    def __init__(
        self,
        task: str = "age",
        hidden_dim: int = 512,
        conv_width: Optional[list[int]] = None,
        dropout: float = 0.1,
        lead: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task = task
        self.hidden_dim = hidden_dim
        self.conv_width = conv_width
        self.dropout = dropout
        self.lead = lead


# ---------------- Submodules ----------------
class CNNLeadEncoder(nn.Module):
    """Encode one lead signal (B,1,T,F)->(B,hid_dim)"""
    def __init__(self, in_ch=1, hid_dim=512, conv_widths=None, dropout=0.1):
        super().__init__()
        chs = [in_ch, 32, 64, 128, 256]
        k = 5
        p = k // 2
        self.hidden_dim = hid_dim
        self.block1 = nn.Sequential(
            nn.Conv2d(chs[0], chs[1], (k, k), padding=(p, p)),
            nn.BatchNorm2d(chs[1]), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(chs[1], chs[2], (k, k), padding=(p, p)),
            nn.BatchNorm2d(chs[2]), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(chs[2], chs[3], (k, k), padding=(p, p)),
            nn.BatchNorm2d(chs[3]), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(chs[3], chs[4], (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(chs[4]), nn.ReLU(inplace=True),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.proj(x)
        return x


# ---------------- HF Model ----------------
class ECGModelHF(PreTrainedModel):
    config_class = ECGConfig

    def __init__(self, config: ECGConfig):
        super().__init__(config)
        self.task = config.task
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.lead = config.lead

        self.encoder = nn.ModuleDict({
            lead: CNNLeadEncoder(1, config.hidden_dim, config.conv_width, config.dropout)
            for lead in _LEAD_ORDER
        })

        enc_dim = config.hidden_dim * len(_LEAD_ORDER)

        if self.task == "age":
            self.head = nn.Sequential(
                nn.Linear(enc_dim, 512), nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512, 1)
            )
            self.loss_fn = F.mse_loss
        elif self.task == "sex":
            self.head = nn.Sequential(
                nn.Linear(enc_dim, 512), nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512, 2)
            )
            self.loss_fn = F.cross_entropy
        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.post_init()

    def forward(self, logmels: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None, **kwargs):
        outs = [self.encoder[lead](x) for lead, x in logmels.items()]
        enc = torch.cat(outs, dim=-1)
        logits = self.head(enc)
        loss = None
        if labels is not None:
            if self.task == "age":
                loss = self.loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss = self.loss_fn(logits, labels.long())
        return {"loss": loss, "logits": logits}
