"""
Train Extra Trees regression models for engine RPM and torque prediction.

The script:
1. Downloads sets A-D from the Hugging Face dataset.
2. Extracts 36 audio features from 2-second chunks.
3. Splits data by source WAV file to prevent data leakage.
4. Trains separate Extra Trees models for RPM and torque.
5. Evaluates on validation and test sets.
6. Saves both models, feature names, and preprocessing configuration to a .pkl file.

Usage:
    python train.py

Optional:
    python train.py --data-dir ./data --output ./models/extra_trees_models.pkl
"""

import argparse
import os
import pickle

import librosa
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID = "rdoerfler/procedural-engine-sounds"

SETS = [
    "A_full_set",
    "B_full_set",
    "C_full_set",
    "D_full_set",
]

CHUNK_DURATION = 2.0
TARGET_SR = 16000
N_MFCC = 13
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset(data_dir):
    """Download only the four required WAV sets from Hugging Face."""

    data_dir = os.path.abspath(data_dir)

    print("=" * 60)
    print("Downloading dataset")
    print("=" * 60)

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[
            "dataset/audio/A_full_set/*.wav",
            "dataset/audio/B_full_set/*.wav",
            "dataset/audio/C_full_set/*.wav",
            "dataset/audio/D_full_set/*.wav",
        ],
        local_dir=data_dir,
    )

    print(f"\nDataset available at: {local_dir}")
    return local_dir


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_feature_names():
    """Return names for the 36 extracted features."""

    return (
        [f"MFCC_{i}_mean" for i in range(1, N_MFCC + 1)]
        + [f"MFCC_{i}_std" for i in range(1, N_MFCC + 1)]
        + [
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_bandwidth_mean",
            "spectral_bandwidth_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "zero_crossing_mean",
            "zero_crossing_std",
            "rms_mean",
            "rms_std",
        ]
    )


def extract_features(dataset_root):
    """
    Extract 36 features from every complete 2-second chunk.

    Channels:
        0-1: stereo engine audio
        2:   RPM annotation, scaled by 10000
        3:   torque annotation, scaled by 1000

    Returns:
        X: feature matrix
        y_rpm: RPM targets
        y_torque: torque targets
        source_files: source WAV ID for every feature row
    """

    audio_root = os.path.join(dataset_root, "dataset", "audio")

    X = []
    y_rpm = []
    y_torque = []
    source_files = []

    print("\n" + "=" * 60)
    print("Extracting features")
    print("=" * 60)

    for set_name in SETS:
        set_path = os.path.join(audio_root, set_name)

        if not os.path.isdir(set_path):
            raise FileNotFoundError(
                f"Expected dataset directory was not found: {set_path}"
            )

        wav_files = sorted(
            f for f in os.listdir(set_path)
            if f.lower().endswith(".wav")
        )

        print(f"\nProcessing {set_name}: {len(wav_files)} files")

        for filename in tqdm(wav_files, desc=set_name):
            file_path = os.path.join(set_path, filename)

            try:
                audio, sr = sf.read(file_path)

                if audio.ndim != 2 or audio.shape[1] != 4:
                    print(f"Skipping unexpected file: {filename}")
                    continue

                # Channels 1-2: stereo engine audio.
                engine_mono = audio[:, :2].mean(axis=1)

                # Channels 3-4: RPM and torque annotations.
                rpm_signal = audio[:, 2] * 10000
                torque_signal = audio[:, 3] * 1000

                chunk_samples = int(CHUNK_DURATION * sr)
                num_chunks = len(engine_mono) // chunk_samples

                for chunk_index in range(num_chunks):
                    start = chunk_index * chunk_samples
                    end = start + chunk_samples

                    audio_chunk = engine_mono[start:end]
                    rpm_chunk = rpm_signal[start:end]
                    torque_chunk = torque_signal[start:end]

                    # Resample only the engine audio to 16 kHz.
                    audio_16k = librosa.resample(
                        audio_chunk,
                        orig_sr=sr,
                        target_sr=TARGET_SR,
                    )

                    # MFCC features: 13 means + 13 standard deviations.
                    mfcc = librosa.feature.mfcc(
                        y=audio_16k,
                        sr=TARGET_SR,
                        n_mfcc=N_MFCC,
                    )

                    mfcc_mean = mfcc.mean(axis=1)
                    mfcc_std = mfcc.std(axis=1)

                    # Additional spectral/audio features.
                    spectral_centroid = librosa.feature.spectral_centroid(
                        y=audio_16k,
                        sr=TARGET_SR,
                    )

                    spectral_bandwidth = librosa.feature.spectral_bandwidth(
                        y=audio_16k,
                        sr=TARGET_SR,
                    )

                    spectral_rolloff = librosa.feature.spectral_rolloff(
                        y=audio_16k,
                        sr=TARGET_SR,
                    )

                    zero_crossing = librosa.feature.zero_crossing_rate(
                        audio_16k
                    )

                    rms = librosa.feature.rms(y=audio_16k)

                    extra_features = np.array([
                        spectral_centroid.mean(),
                        spectral_centroid.std(),
                        spectral_bandwidth.mean(),
                        spectral_bandwidth.std(),
                        spectral_rolloff.mean(),
                        spectral_rolloff.std(),
                        zero_crossing.mean(),
                        zero_crossing.std(),
                        rms.mean(),
                        rms.std(),
                    ])

                    features = np.concatenate([
                        mfcc_mean,
                        mfcc_std,
                        extra_features,
                    ])

                    X.append(features)
                    y_rpm.append(rpm_chunk.mean())
                    y_torque.append(torque_chunk.mean())
                    source_files.append(f"{set_name}/{filename}")

            except Exception as exc:
                print(f"Error processing {filename}: {exc}")

    if not X:
        raise RuntimeError("No feature samples were extracted.")

    X = np.asarray(X, dtype=np.float32)
    y_rpm = np.asarray(y_rpm, dtype=np.float32)
    y_torque = np.asarray(y_torque, dtype=np.float32)
    source_files = np.asarray(source_files)

    print("\nFeature extraction complete.")
    print(f"X shape: {X.shape}")
    print(f"RPM shape: {y_rpm.shape}")
    print(f"Torque shape: {y_torque.shape}")
    print(f"Unique WAV files: {len(np.unique(source_files))}")

    return X, y_rpm, y_torque, source_files


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(X, y_rpm, y_torque, source_files):
    """
    Split into 70% train, 15% validation, 15% test.

    GroupShuffleSplit keeps all chunks from one WAV file in the same split,
    preventing leakage between train/validation/test sets.
    """

    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        gss1.split(X, y_rpm, groups=source_files)
    )

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    val_relative_idx, test_relative_idx = next(
        gss2.split(
            X[temp_idx],
            y_rpm[temp_idx],
            groups=source_files[temp_idx],
        )
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    # Verify there is no source-file overlap.
    train_files = set(source_files[train_idx])
    val_files = set(source_files[val_idx])
    test_files = set(source_files[test_idx])

    assert not train_files.intersection(val_files)
    assert not train_files.intersection(test_files)
    assert not val_files.intersection(test_files)

    print("\n" + "=" * 60)
    print("Dataset split")
    print("=" * 60)

    print(f"Training:   {len(train_idx):5d} samples | "
          f"{len(train_files):4d} WAV files")
    print(f"Validation: {len(val_idx):5d} samples | "
          f"{len(val_files):4d} WAV files")
    print(f"Testing:    {len(test_idx):5d} samples | "
          f"{len(test_files):4d} WAV files")

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_models(X_train, y_rpm_train, y_torque_train):
    """Train separate Extra Trees regressors for RPM and torque."""

    print("\n" + "=" * 60)
    print("Training models")
    print("=" * 60)

    rpm_model = ExtraTreesRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    torque_model = ExtraTreesRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training RPM model...")
    rpm_model.fit(X_train, y_rpm_train)

    print("Training torque model...")
    torque_model.fit(X_train, y_torque_train)

    return rpm_model, torque_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X, y, target_name):
    """Calculate and print MAE, RMSE and R²."""

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    print(f"\n{target_name}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ---------------------------------------------------------------------------
# Save models
# ---------------------------------------------------------------------------

def save_models(output_path, rpm_model, torque_model, feature_names):
    """Save both trained models and metadata into one pickle file."""

    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model_package = {
        "rpm_model": rpm_model,
        "torque_model": torque_model,
        "feature_names": feature_names,
        "config": {
            "chunk_duration": CHUNK_DURATION,
            "target_sr": TARGET_SR,
            "n_mfcc": N_MFCC,
            "feature_count": len(feature_names),
            "sets": SETS,
        },
    }

    with open(output_path, "wb") as file:
        pickle.dump(model_package, file)

    print(f"\nModels saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train engine RPM and torque prediction models."
    )

    parser.add_argument(
        "--data-dir",
        default="./data/procedural-engine-sounds",
        help="Directory where the Hugging Face dataset will be downloaded.",
    )

    parser.add_argument(
        "--output",
        default="./models/extra_trees_models.pkl",
        help="Path for the generated pickle model file.",
    )

    args = parser.parse_args()

    # Download dataset.
    dataset_root = download_dataset(args.data_dir)

    # Extract the 36-feature dataset.
    X, y_rpm, y_torque, source_files = extract_features(dataset_root)

    # Split by source WAV file.
    train_idx, val_idx, test_idx = split_dataset(
        X,
        y_rpm,
        y_torque,
        source_files,
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_rpm_train = y_rpm[train_idx]
    y_rpm_val = y_rpm[val_idx]
    y_rpm_test = y_rpm[test_idx]

    y_torque_train = y_torque[train_idx]
    y_torque_val = y_torque[val_idx]
    y_torque_test = y_torque[test_idx]

    # Train models.
    rpm_model, torque_model = train_models(
        X_train,
        y_rpm_train,
        y_torque_train,
    )

    # Validation results.
    print("\n" + "=" * 60)
    print("Validation results")
    print("=" * 60)

    evaluate_model(rpm_model, X_val, y_rpm_val, "RPM")
    evaluate_model(torque_model, X_val, y_torque_val, "Torque")

    # Final test results.
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    evaluate_model(rpm_model, X_test, y_rpm_test, "RPM")
    evaluate_model(torque_model, X_test, y_torque_test, "Torque")

    # Feature importance.
    feature_names = get_feature_names()

    print("\n" + "=" * 60)
    print("Top RPM features")
    print("=" * 60)

    rpm_importance = sorted(
        zip(feature_names, rpm_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, importance in rpm_importance[:10]:
        print(f"{feature:30s} {importance:.6f}")

    print("\n" + "=" * 60)
    print("Top torque features")
    print("=" * 60)

    torque_importance = sorted(
        zip(feature_names, torque_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, importance in torque_importance[:10]:
        print(f"{feature:30s} {importance:.6f}")

    # Save final artifact.
    save_models(
        args.output,
        rpm_model,
        torque_model,
        feature_names,
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
