"""
Inspect saved model artifacts without assuming the audio preprocessing.
Run: python inspect_models.py
Only load pickle/joblib files you trust; loading a pickle can execute code.
"""
from pathlib import Path
import pickle
import pprint
import joblib

ROOT = Path(__file__).resolve().parent

def load(path):
    try:
        return joblib.load(path)
    except Exception as first:
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as second:
            print(f"Could not load {path.name}: {first}; fallback: {second}")
            return None

def describe_object(name, obj, depth=0):
    pad = "  " * depth
    print(f"{pad}{name}: type={type(obj).__module__}.{type(obj).__name__}")
    for attr in ("n_features_in_", "n_outputs_", "classes_", "feature_names_in_"):
        if hasattr(obj, attr):
            try:
                print(f"{pad}  {attr} = {getattr(obj, attr)!r}")
            except Exception:
                pass
    if hasattr(obj, "steps"):
        print(f"{pad}  pipeline steps:")
        for step_name, step in obj.steps:
            describe_object(step_name, step, depth + 2)

def main():
    for filename in ("safety_model.pkl", "extra_trees_models.pkl"):
        path = ROOT / filename
        print("\n" + "=" * 72)
        print(f"FILE: {path}")
        if not path.exists():
            print("NOT FOUND. Place this file beside inspect_models.py.")
            continue
        obj = load(path)
        if obj is None:
            continue
        describe_object(filename, obj)
        if isinstance(obj, dict):
            print("Dictionary keys:", list(obj.keys()))
            for key, value in obj.items():
                if key == "config":
                    print("config:")
                    pprint.pp(value, sort_dicts=False)
                elif key in ("feature_names", "rpm_model", "torque_model", "model", "classifier", "pipeline"):
                    describe_object(key, value, 1)
                elif isinstance(value, (str, int, float, bool, list, tuple)):
                    print(f"  {key} = {value!r}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        elif hasattr(obj, "get_params"):
            try:
                print("Estimator parameters:")
                pprint.pp(obj.get_params(deep=False), sort_dicts=False)
            except Exception:
                pass
    print("\nSend the full output back, especially feature_names, config, classes_,")
    print("n_features_in_, and any pipeline steps. Do not upload private data.")

if __name__ == "__main__":
    main()
