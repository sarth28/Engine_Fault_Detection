# Engine RPM & Torque Prediction

## Train the models

```bash
pip install -r requirements.txt
python train.py
```

The script downloads only sets `A_full_set` through `D_full_set` from the
`rdoerfler/procedural-engine-sounds` Hugging Face dataset, extracts 36 audio
features from 2-second chunks, trains two Extra Trees regression models, and
saves:

```text
models/extra_trees_models.pkl
```

You can change the dataset and output locations:

```bash
python train.py --data-dir ./data/procedural-engine-sounds --output ./models/extra_trees_models.pkl
```

The pickle contains:

- `rpm_model`
- `torque_model`
- `feature_names`
- `config`

The train/validation/test split is performed by source WAV file rather than
individual chunks, preventing chunks from the same recording from appearing
in multiple splits.

## Run the app
```bash
streamlit run app.py
```
