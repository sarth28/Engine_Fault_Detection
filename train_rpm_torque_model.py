
import os
import pickle
import warnings

import numpy as np
import soundfile as sf
import librosa

from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ID = "rdoerfler/procedural-engine-sounds"

MODEL_FILE = "extra_trees_models.pkl"

SAMPLE_RATE = 16000
CHUNK_DURATION = 2
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

RANDOM_STATE = 42


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    audio = np.asarray(audio, dtype=np.float32)

    # MFCC features
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE
    )[0]

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=audio
    )[0]

    rms = librosa.feature.rms(
        y=audio
    )[0]

    spectral_features = [
        np.mean(spectral_centroid),
        np.std(spectral_centroid),

        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),

        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),

        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),

        np.mean(rms),
        np.std(rms)
    ]

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        spectral_features
    ])

    return features


# ============================================================
# TRAIN RPM AND TORQUE MODELS
# ============================================================

def train_models():

    print("\n" + "=" * 60)
    print("RPM AND TORQUE REGRESSION TRAINING")
    print("=" * 60)

    if os.path.exists(MODEL_FILE):

        print(f"\n{MODEL_FILE} already exists.")
        print("Skipping model training.")

        return

    # --------------------------------------------------------
    # DOWNLOAD DATASET
    # --------------------------------------------------------

    print("\nDownloading procedural engine sounds dataset...")

    dataset_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        allow_patterns=[
            "dataset/audio/A_full_set/*.wav",
            "dataset/audio/B_full_set/*.wav",
            "dataset/audio/C_full_set/*.wav",
            "dataset/audio/D_full_set/*.wav"
        ]
    )

    dataset_path = Path(dataset_path)

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    X = []
    rpm_values = []
    torque_values = []
    groups = []

    dataset_folders = [
        "A_full_set",
        "B_full_set",
        "C_full_set",
        "D_full_set"
    ]

    print("\nExtracting audio features...")

    for folder in dataset_folders:

        audio_folder = (
            dataset_path / "dataset" / "audio" / folder
        )

        wav_files = list(audio_folder.glob("*.wav"))

        print(f"\n{folder}: {len(wav_files)} WAV files")

        for wav_file in tqdm(wav_files):

            try:

                audio_data, sr = sf.read(
                    str(wav_file),
                    dtype="float32"
                )

                # Expected format: 4 channels
                if audio_data.ndim != 2 or audio_data.shape[1] < 4:
                    continue

                # Resample if necessary
                if sr != SAMPLE_RATE:

                    resampled_channels = []

                    for channel in range(4):

                        resampled = librosa.resample(
                            audio_data[:, channel],
                            orig_sr=sr,
                            target_sr=SAMPLE_RATE
                        )

                        resampled_channels.append(resampled)

                    min_length = min(
                        len(channel)
                        for channel in resampled_channels
                    )

                    audio_data = np.column_stack([
                        channel[:min_length]
                        for channel in resampled_channels
                    ])

                # First two channels contain engine audio
                engine_audio = np.mean(
                    audio_data[:, 0:2],
                    axis=1
                )

                # Annotation channels
                rpm_annotation = audio_data[:, 2] * 10000
                torque_annotation = audio_data[:, 3] * 1000

                # Divide audio into 2-second chunks
                number_of_chunks = (
                    len(engine_audio) // CHUNK_SIZE
                )

                for chunk_index in range(number_of_chunks):

                    start = chunk_index * CHUNK_SIZE
                    end = start + CHUNK_SIZE

                    audio_chunk = engine_audio[start:end]

                    rpm_chunk = rpm_annotation[start:end]
                    torque_chunk = torque_annotation[start:end]

                    # Extract features
                    features = extract_features(audio_chunk)

                    rpm = np.mean(rpm_chunk)
                    torque = np.mean(torque_chunk)

                    if not np.all(np.isfinite(features)):
                        continue

                    if not np.isfinite(rpm) or not np.isfinite(torque):
                        continue

                    X.append(features)
                    rpm_values.append(rpm)
                    torque_values.append(torque)

                    # Keep chunks from the same file together
                    groups.append(
                        f"{folder}/{wav_file.name}"
                    )

            except Exception as e:

                print(f"\nSkipping {wav_file.name}: {e}")

    X = np.asarray(X)
    rpm_values = np.asarray(rpm_values)
    torque_values = np.asarray(torque_values)
    groups = np.asarray(groups)

    print("\nFeature extraction completed.")

    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")

    if len(X) == 0:
        raise ValueError("No training samples were extracted.")

    # --------------------------------------------------------
    # GROUP-BASED TRAIN-TEST SPLIT
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(X, rpm_values, groups)
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_rpm_train = rpm_values[train_indices]
    y_rpm_test = rpm_values[test_indices]

    y_torque_train = torque_values[train_indices]
    y_torque_test = torque_values[test_indices]

    # --------------------------------------------------------
    # TRAIN RPM MODEL
    # --------------------------------------------------------

    print("\nTraining RPM Extra Trees model...")

    rpm_model = ExtraTreesRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rpm_model.fit(
        X_train,
        y_rpm_train
    )

    rpm_predictions = rpm_model.predict(X_test)

    rpm_mae = mean_absolute_error(
        y_rpm_test,
        rpm_predictions
    )

    rpm_rmse = np.sqrt(
        mean_squared_error(
            y_rpm_test,
            rpm_predictions
        )
    )

    rpm_r2 = r2_score(
        y_rpm_test,
        rpm_predictions
    )

    print("\nRPM Model Results")
    print(f"MAE:  {rpm_mae:.2f}")
    print(f"RMSE: {rpm_rmse:.2f}")
    print(f"R2:   {rpm_r2:.4f}")

    # --------------------------------------------------------
    # TRAIN TORQUE MODEL
    # --------------------------------------------------------

    print("\nTraining Torque Extra Trees model...")

    torque_model = ExtraTreesRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    torque_model.fit(
        X_train,
        y_torque_train
    )

    torque_predictions = torque_model.predict(X_test)

    torque_mae = mean_absolute_error(
        y_torque_test,
        torque_predictions
    )

    torque_rmse = np.sqrt(
        mean_squared_error(
            y_torque_test,
            torque_predictions
        )
    )

    torque_r2 = r2_score(
        y_torque_test,
        torque_predictions
    )

    print("\nTorque Model Results")
    print(f"MAE:  {torque_mae:.2f}")
    print(f"RMSE: {torque_rmse:.2f}")
    print(f"R2:   {torque_r2:.4f}")

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    feature_names = []

    for i in range(1, 14):
        feature_names.append(f"mfcc_{i}_mean")

    for i in range(1, 14):
        feature_names.append(f"mfcc_{i}_std")

    feature_names.extend([
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "rms_mean",
        "rms_std"
    ])

    model_data = {
        "rpm_model": rpm_model,
        "torque_model": torque_model,
        "feature_names": feature_names,
        "config": {
            "sample_rate": SAMPLE_RATE,
            "chunk_duration": CHUNK_DURATION,
            "n_mfcc": 13,
            "total_features": 36
        }
    }

    with open(MODEL_FILE, "wb") as file:

        pickle.dump(
            model_data,
            file
        )

    print(f"\nModels saved successfully: {MODEL_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_models()

    print("\n" + "=" * 60)
    print("REGRESSION MODEL TRAINING COMPLETED")
    print("=" * 60)
