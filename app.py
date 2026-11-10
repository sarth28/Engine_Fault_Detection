from __future__ import annotations
from pathlib import Path
import pickle
import concurrent.futures
import joblib
import numpy as np
import streamlit as st
from audio_utils import load_audio, extract_named_features

ROOT = Path(__file__).resolve().parent
SAFETY_PATH = ROOT / "safety_model.pkl"
REGRESSION_PATH = ROOT / "extra_trees_models.pkl"

st.set_page_config(page_title="Diesel Engine Analyzer", page_icon="🎛️", layout="wide")

@st.cache_resource
def load_artifact(path_string):
    path = Path(path_string)
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)

def get_regression_parts(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("Expected extra_trees_models.pkl to contain a dictionary.")
    required = ("rpm_model", "torque_model", "feature_names")
    missing = [k for k in required if k not in artifact]
    if missing:
        raise ValueError("Regression artifact is missing keys: " + ", ".join(missing))
    feature_names = artifact["feature_names"]
    if hasattr(feature_names, "tolist"):
        feature_names = feature_names.tolist()
    if not isinstance(feature_names, (list, tuple)) or not feature_names:
        raise ValueError("'feature_names' must be a non-empty list or array.")
    return artifact["rpm_model"], artifact["torque_model"], list(feature_names), artifact.get("config", {})

def predict_regression(y, sr):
    artifact = load_artifact(str(REGRESSION_PATH))
    rpm_model, torque_model, feature_names, config = get_regression_parts(artifact)
    # If the saved training config explicitly states a sample rate, honor it.
    target_sr = None
    if isinstance(config, dict):
        for key in ("sample_rate", "sr", "sampling_rate", "target_sr"):
            if config.get(key):
                target_sr = int(config[key])
                break
    if target_sr and target_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    X = extract_named_features(y, sr, feature_names)
    expected = getattr(rpm_model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(f"Extractor produced {X.shape[1]} features but RPM model expects {expected}.")
    rpm = float(np.asarray(rpm_model.predict(X)).reshape(-1)[0])
    torque = float(np.asarray(torque_model.predict(X)).reshape(-1)[0])
    if not np.isfinite(rpm) or not np.isfinite(torque):
        raise ValueError("The regression model returned a non-finite value.")
    return {"rpm": rpm, "torque": torque, "features_used": len(feature_names), "sample_rate": sr}

def predict_safety(y, sr):
    """Predict safety using the same 13 mean MFCCs used during training."""
    artifact = load_artifact(str(SAFETY_PATH))
    if not hasattr(artifact, "predict"):
        raise ValueError("safety_model.pkl does not expose predict().")

    # Match the notebook exactly: extract 13 MFCCs at the WAV's original
    # sample rate, then average each coefficient over all audio frames.
    import librosa
    mfccs = librosa.feature.mfcc(y=np.asarray(y, dtype=np.float32), sr=int(sr), n_mfcc=13)
    mfcc_means = np.mean(mfccs.T, axis=0).reshape(1, -1)

    expected = getattr(artifact, "n_features_in_", None)
    if expected is not None and mfcc_means.shape[1] != int(expected):
        raise ValueError(
            f"MFCC extraction produced {mfcc_means.shape[1]} features, "
            f"but the classifier expects {expected}."
        )

    raw_prediction = int(np.asarray(artifact.predict(mfcc_means)).reshape(-1)[0])
    # Training notebook mapping: 1 = Safe/Normal, 0 = Anomaly.
    label_map = {1: "Safe", 0: "Anomaly"}
    if raw_prediction not in label_map:
        raise ValueError(f"Unexpected safety class returned by model: {raw_prediction}")
    result = {"label": label_map[raw_prediction], "confidence": None}

    if hasattr(artifact, "predict_proba"):
        try:
            probabilities = artifact.predict_proba(mfcc_means)[0]
            classes = list(getattr(artifact, "classes_", []))
            if raw_prediction in classes:
                result["confidence"] = float(probabilities[classes.index(raw_prediction)])
        except Exception:
            pass
    return result

st.title("Diesel Engine Audio Analyzer")
st.caption("Upload one engine recording to run safety classification and RPM/torque estimation.")

#with st.sidebar:
#    st.info("Model predictions are only reliable when inference preprocessing matches training.")

if not SAFETY_PATH.exists() or not REGRESSION_PATH.exists():
    st.warning("Place safety_model.pkl and extra_trees_models.pkl beside app.py, then restart or refresh the app.")

uploaded = st.file_uploader("Upload diesel engine audio", type=["wav", "flac", "ogg", "aiff", "aif"])
if uploaded:
    st.audio(uploaded)
    try:
        y, sr = load_audio(uploaded)
    except Exception as exc:
        st.error(f"Could not read audio: {exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{len(y) / sr:.2f} s")
    c2.metric("Original sample rate", f"{sr:,} Hz")
    c3.metric("Audio samples", f"{len(y):,}")

    if st.button("Analyze engine", type="primary", use_container_width=True):
        if not SAFETY_PATH.exists() or not REGRESSION_PATH.exists():
            st.error("Both model files must be present before analysis.")
            st.stop()

        with st.spinner("Running both models concurrently..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                safety_future = pool.submit(predict_safety, y.copy(), sr)
                regression_future = pool.submit(predict_regression, y.copy(), sr)
                safety_result = None
                regression_result = None
                safety_error = None
                regression_error = None
                try:
                    safety_result = safety_future.result()
                except Exception as exc:
                    safety_error = str(exc)
                try:
                    regression_result = regression_future.result()
                except Exception as exc:
                    regression_error = str(exc)

        st.subheader("Analysis results")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Safety classification")
            if safety_result:
                label = safety_result["label"]
                if label.lower() == "safe":
                    st.success(f"Result: {label}")
                elif label.lower() == "anomaly":
                    st.error(f"Result: {label}")
                else:
                    st.info(f"Result: {label}")
                if safety_result["confidence"] is not None:
                    st.metric("Model confidence", f"{safety_result['confidence'] * 100:.1f}%")
            else:
                st.warning("Safety result unavailable")
                st.code(safety_error or "Unknown error")
        with right:
            st.markdown("#### RPM and torque estimation")
            if regression_result:
                st.metric("Predicted RPM", f"{regression_result['rpm']:,.2f} RPM")
                st.metric("Predicted torque", f"{regression_result['torque']:,.2f} Nm")
                st.caption(f"Features supplied: {regression_result['features_used']}; sample rate used: {regression_result['sample_rate']:,} Hz")
            else:
                st.warning("RPM/torque results unavailable")
                st.code(regression_error or "Unknown error")

        st.caption("This is a research prototype, not a certified engine safety system. Validate predictions against labeled recordings and real operating measurements.")
else:
    st.info("Choose an audio file to begin.")
