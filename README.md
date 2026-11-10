# Diesel Engine Audio Analyzer

A Streamlit-based application that analyzes diesel engine audio recordings using machine learning.

### Features
- **Safety Classification:** Classifies engine audio as Safe or Anomaly using a Random Forest classifier.
- **RPM Estimation:** Predicts engine RPM using an Extra Trees Regressor.
- **Torque Estimation:** Predicts engine torque using an Extra Trees Regressor.
- **Concurrent Prediction:** Runs both models simultaneously.

### Project Files

| File | Description |
|---|---|
| `app.py` | Streamlit application for audio upload and predictions. |
| `audio_utils.py` | Handles audio loading and feature extraction for model predictions. |
| `train_rpm_torque_model.py` | Trains and saves the RPM and torque regression models. |
| `inspect_models.py` | Inspects saved model artifacts, including model types, features and configurations. |
| `safety_model.pkl` | Pre-trained Random Forest safety classifier (included in the repository). |
| `extra_trees_models.pkl` | Saved RPM and torque regression models. |

### Installation

```bash
pip install -r requirements.txt
```

### Model Training

The safety classification model is already included in the repository.

To generate the RPM and torque models, run:

```bash
python train_rpm_torque_model.py
```

This downloads the required dataset, extracts audio features, trains both regression models and saves them as `extra_trees_models.pkl`.

To inspect the saved models, run:

```bash
python inspect_models.py
```

### Run the Application

```bash
streamlit run app.py
```

Upload a supported audio file (`.wav`, `.flac`, `.ogg`, `.aiff`) and click **Analyze Engine** to view the predictions.

**Note:** This is a research prototype and not a certified engine safety system.
