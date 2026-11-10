"""Audio loading and conservative feature extraction helpers.

Important:
- RPM/torque feature names and config must match the training notebook.
- This module supports only explicitly named common features. Unknown feature
  names raise an error instead of silently inventing incompatible inputs.
- Safety-model waveform preprocessing is intentionally not guessed.
"""
from __future__ import annotations
import re
import numpy as np
import soundfile as sf
import librosa

SUPPORTED_PREFIXES = {
    "mfcc_mean", "mfcc_std",
}

def load_audio(file_obj, target_sr=None):
    """Load audio to mono float32. Resampling happens only when target_sr is set."""
    file_obj.seek(0)
    y, sr = sf.read(file_obj, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if y.size == 0:
        raise ValueError("The uploaded audio file is empty.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio contains NaN or infinite values.")
    if target_sr is not None and int(target_sr) != int(sr):
        y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
        sr = int(target_sr)
    return y.astype(np.float32), int(sr)

def _spectral_values(y, sr):
    if len(y) < 16:
        raise ValueError("Audio is too short to extract audio features.")
    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(len(y))))))
    hop = max(1, n_fft // 4)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=min(2048, max(64, 2**int(np.floor(np.log2(len(y)))))))[0]
    rms = librosa.feature.rms(S=S)[0]
    return {
        "duration": float(len(y) / sr),
        "sample_rate": float(sr),
        "amplitude_mean": float(np.mean(y)),
        "amplitude_std": float(np.std(y)),
        "amplitude_rms": float(np.sqrt(np.mean(y**2))),
        "amplitude_max": float(np.max(np.abs(y))),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "zero_crossing_rate_std": float(np.std(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }

def extract_named_features(y, sr, feature_names):
    """Extract supported features in the exact order requested by feature_names."""
    base = _spectral_values(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        base[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        base[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    aliases = {
        "duration_seconds": "duration",
        "duration_sec": "duration",
        "rms": "amplitude_rms",
        "mean_amplitude": "amplitude_mean",
        "std_amplitude": "amplitude_std",
        "max_amplitude": "amplitude_max",
        "zcr_mean": "zero_crossing_rate_mean",
        "zcr_std": "zero_crossing_rate_std",
        "zero_crossing_mean": "zero_crossing_rate_mean",
        "zero_crossing_std": "zero_crossing_rate_std",
        "spectral_centroid": "spectral_centroid_mean",
        "spectral_bandwidth": "spectral_bandwidth_mean",
        "spectral_rolloff": "spectral_rolloff_mean",
        "spectral_flatness": "spectral_flatness_mean",
    }
    values = []
    missing = []
    for raw_name in feature_names:
        name = str(raw_name).strip()
        canonical = aliases.get(name.lower(), name.lower())
        if canonical in base:
            values.append(base[canonical])
        else:
            # Recognize MFCC naming patterns such as mfcc_1_mean / mfcc1_std.
            m = re.fullmatch(r"mfcc[_-]?(\d+)[_-](mean|std)", canonical)
            if m:
                key = f"mfcc_{int(m.group(1))}_{m.group(2)}"
                if key in base:
                    values.append(base[key])
                else:
                    missing.append(name)
            else:
                missing.append(name)
    if missing:
        raise ValueError(
            "The RPM/torque model requests feature names this generic extractor "
            "cannot safely reproduce: " + ", ".join(missing) +
            ". Use the exact feature-extraction code from the training notebook."
        )
    return np.asarray(values, dtype=np.float64).reshape(1, -1)
