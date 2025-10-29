from typing import Dict, Tuple, List, Any
from pathlib import Path

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, iirnotch

from ..interfaces.protocol import PipelineProtocol

# ---------------- Constant ----------------
_X_SECOND_PER_GRID = 0.04 # 25 mm/s -> 1 small grid = 0.04s
_Y_MV_PER_GRID = 0.1 # 10 mm/mV -> 1 small grid = 0.1mV
_SECOND_PER_LEAD = 2.5 # Standard ECG lead duration = 2 seconds
_MV_PER_LEAD = 5.0 # Standard ECG lead amplitude = 5 mV
_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# ---------------- ECG Data ----------------
class ECGData:
    def __init__(self, data_config: str | Path):
        self.data_config = pd.read_csv(data_config)
        self.point_data: List[pd.DataFrame]
        self.point_meta: List[dict]
        self.point_data = []
        self.point_meta = []
        self.point_data = self.get_point_data()
        self.lead_data: Dict[str, List[pd.DataFrame]] = self.get_lead_data()
        self.lead_info: Dict[str, pd.DataFrame] = self.get_lead_info()
        self.sample: List[pd.DataFrame]
        self.sample_meta: List[dict]
        self.sample = self.get_sample()

    def get_point_data(self):
        point_data = []
        point_meta = []
        error_counts = 0

        for _, row in tqdm(self.data_config.iterrows(), total=len(self.data_config), desc="Loading CSVs"):
            file_path = row["file"]
            try:
                data = pd.read_csv(file_path)
                point_data.append(data)
                meta = {
                    "file": file_path,
                    "age": row.get("age", None),
                    "sex": row.get("sex", None),
                    "name": row.get("name", None),
                }
                point_meta.append(meta)
            except Exception:
                error_counts += 1

        cleaned_points = []
        cleaned_meta = []
        for i, point in enumerate(tqdm(point_data, desc="Dropping outliers")):
            cleaned_leads = []
            for lead in _LEAD_ORDER:
                lead_sub = point.loc[point["lead"] == lead]
                if not lead_sub.empty:
                    lead_sub = self.drop_outlier(lead_sub, cols=["x", "y"], k=0.2)
                    cleaned_leads.append(lead_sub)
            if cleaned_leads:
                cleaned_points.append(pd.concat(cleaned_leads, ignore_index=True))
                cleaned_meta.append(point_meta[i])

        self.point_meta = cleaned_meta
        print(f"Total files with errors: {error_counts}")
        return cleaned_points

    def drop_outlier(self, df: pd.DataFrame, cols: List[str], k: float) -> pd.DataFrame:
        clean_df = df.copy()
        for col in cols:
            q_low, q_high = clean_df[col].quantile([0.01, 0.99])
            rng = q_high - q_low
            clean_df = clean_df[(clean_df[col] >= q_low - k * rng) & (clean_df[col] <= q_high + k * rng)]
        return clean_df

    def get_sample(self):
        sample = []
        sample_meta = []
        for i, df in enumerate(self.point_data):
            keep = True
            for lead in _LEAD_ORDER:
                sub = df.loc[df["lead"] == lead]
                if sub.empty:
                    keep = False
                    break
                info_dict = self.get_info(sub)
                lead_info = self.lead_info[lead]
                for info, value in info_dict.items():
                    q_low, q_high = lead_info[info].quantile([0.01, 0.99])
                    rng = q_high - q_low
                    if value < q_low - 0.05 * rng or value > q_high + 0.05 * rng:
                        keep = False
                        break
                if not keep:
                    break
            if keep:
                sample.append(df)
                sample_meta.append(self.point_meta[i])

        self.sample_meta = sample_meta
        print(f"Total samples after filtering: {len(sample)}")
        return sample
    
    def get_lead_data(self):
        lead_data = {lead: [] for lead in _LEAD_ORDER}
        for data in tqdm(self.point_data, desc="Organizing lead data"):
            for lead in _LEAD_ORDER:
                lead_sub = data.loc[data["lead"] == lead]
                lead_sub = lead_sub.drop(columns=["lead"]).reset_index(drop=True)
                lead_data[lead].append(lead_sub)
        return lead_data
    
    def get_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        mean_x = df["x"].mean()
        mean_y = df["y"].mean()
        range_x = df["x"].max() - df["x"].min()
        range_y = df["y"].max() - df["y"].min()
        min_x = df["x"].min()
        max_x = df["x"].max()
        min_y = df["y"].min()
        max_y = df["y"].max()
        return {
            "mean_x": mean_x,
            "mean_y": mean_y,
            "range_x": range_x,
            "range_y": range_y,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y
        }
    
    def get_lead_info(self):
        lead_info = {}
        for lead, lead_sub in tqdm(self.lead_data.items(), desc="Calculating lead info"):
            info_df = []
            for df in lead_sub:
                info_df.append(self.get_info(df))
            lead_info[lead] = pd.DataFrame(info_df)
        return lead_info
        
    
# ---------------- Helper ----------------
def zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    if sd < 1e-12:
        sd = 1
    return (x - mu) / sd

def butter_bandpass(sig: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def apply_notch(sig: np.ndarray, fs: float, f0: float, q: float) -> np.ndarray:
    if f0 <= 0:
        return sig
    b, a = iirnotch(w0=f0/(fs/2), Q=q)
    return filtfilt(b, a, sig)


def waveform_to_logmel(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64
) -> torch.Tensor:
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    mel = MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)
    log_mel = AmplitudeToDB()(mel)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel

class PreprocessECGPipeline(PipelineProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        self.ecg_data = ECGData(self.cfg.pipeline.src)
        self.waveforms: List[Dict[str, np.ndarray]] = []
        self.logmels: List[Dict[str, torch.Tensor]] = []

        self.save_dir = Path(self.cfg.pipeline.save)
        self.waveforms_dir = self.save_dir / "waveforms"
        self.logmels_dir = self.save_dir / "logmel"
        self.waveforms_dir.mkdir(parents=True, exist_ok=True)
        self.logmels_dir.mkdir(parents=True, exist_ok=True)

        self.base_name = getattr(self.cfg.pipeline, "base_name", None)

        self.lead_rows: Dict[str, List[dict]] = {ld: [] for ld in _LEAD_ORDER}

    def run(self):
        fs = self.cfg.pipeline.target_fs
        logmel_sr = self.cfg.pipeline.logmel_sr
        n_fft = self.cfg.pipeline.n_fft
        hop = self.cfg.pipeline.hop_length
        n_mels = self.cfg.pipeline.n_mels

        for idx, sample in enumerate(self.ecg_data.sample):
            meta_in = self.ecg_data.sample_meta[idx] if idx < len(self.ecg_data.sample_meta) else {}
            age = meta_in.get("age", None)
            sex = meta_in.get("sex", None)
            name = meta_in.get("name", None)

            basename = (self.base_name or name or f"sample_{idx:04d}")

            lead_waveforms = {}
            for lead in _LEAD_ORDER:
                lead_sub = sample.loc[sample["lead"] == lead]
                if not lead_sub.empty:
                    y = self.get_waveform(lead_sub)  # numpy 1D
                    lead_waveforms[lead] = y

            if not lead_waveforms:
                continue

            peak = max(np.max(np.abs(y)) for y in lead_waveforms.values())
            peak = peak if peak > 0 else 1.0

            for lead in [ld for ld in _LEAD_ORDER if ld in lead_waveforms]:
                y = lead_waveforms[lead]
                y_norm = (y / peak).astype(np.float32)

                wav_path = self.waveforms_dir / f"{basename}_{lead}.wav"
                wav_int16 = (np.clip(y_norm, -1.0, 1.0) * 32767).astype(np.int16)
                wavfile.write(str(wav_path), fs, wav_int16)

                wav_tensor = torch.from_numpy(y_norm).float().unsqueeze(0)  # [1, T]
                if fs != logmel_sr:
                    wav_tensor = torchaudio.functional.resample(wav_tensor, fs, logmel_sr)

                mel_tf = MelSpectrogram(sample_rate=logmel_sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
                mel = mel_tf(wav_tensor)                     # [1, n_mels, time]
                logmel = AmplitudeToDB()(mel)
                logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

                pt_path = self.logmels_dir / f"{basename}_{lead}.pt"
                torch.save(logmel, str(pt_path))

                self.lead_rows[lead].append({
                    "waveform": str(wav_path),
                    "logmel": str(pt_path),
                    "age": age,
                    "sex": sex,
                })

        for lead, rows in self.lead_rows.items():
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["waveform", "logmel", "age", "sex"])
            csv_path = self.save_dir / f"data_config_lead_{lead}.csv"
            df.to_csv(csv_path, index=False)
        
        # Create data config
        data_config = []
        for lead in _LEAD_ORDER:
            data_config.append({
                "lead": lead,
                "src": str(self.save_dir / f"data_config_lead_{lead}.csv")
            })
        data_config_df = pd.DataFrame(data_config, columns=["lead", "src"])
        data_config_path = self.save_dir / "data_config.csv"
        data_config_df.to_csv(data_config_path, index=False)
    
    def get_waveform(self, df: pd.DataFrame):
        def interpolate_uniform(t: np.ndarray, y: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
            order = np.argsort(t)
            t = t[order]
            y = y[order]
            t_uniform = np.arange(t[0], t[-1], 1.0/fs)
            y_uniform = np.interp(t_uniform, t, y)
            return t_uniform, y_uniform

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        
        x = (x - x.min()) / (x.max() - x.min()) * _SECOND_PER_LEAD
        
        x, y = interpolate_uniform(x, y, self.cfg.pipeline.target_fs)

        y = butter_bandpass(y, self.cfg.pipeline.target_fs, self.cfg.pipeline.hp_cutoff_hz, self.cfg.pipeline.lp_cutoff_hz)
        y = apply_notch(y, self.cfg.pipeline.target_fs, self.cfg.pipeline.notch_hz, self.cfg.pipeline.q_factor)
        y = zscore(y)
        return y