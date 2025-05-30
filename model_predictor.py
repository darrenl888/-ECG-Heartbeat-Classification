# model_predictor.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ecg_processing_enhanced import prepare_input_for_model

# === Load models ===
model_mitbih = tf.keras.models.load_model("models/mitbih_model.h5", compile=False)
model_ptbdb = tf.keras.models.load_model("models/ptbdb_model.h5", compile=False)

# === Class Labels ===
mitbih_labels = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
ptbdb_labels = ["Normal", "Abnormal"]

def ecg_predictor_both_models(input_ecg, original_fs):
    """
    Predict input through BOTH models.
    Accepts both full recordings or isolated beats (single or batch).
    Returns the number of beats extracted.
    """
    try:
        beats = prepare_input_for_model(input_ecg, original_fs=original_fs)
    except ValueError as e:
        print(f"🚨 Preprocessing failed: {e}")
        return 0

    if beats.shape[0] == 0:
        print("\n🚨 No valid beats extracted. Severe tachycardia or signal abnormality suspected.\n")
        return 0

    print(f"✅ Number of heartbeats extracted: {beats.shape[0]}")

    preds_mitbih = model_mitbih.predict(beats, verbose=0)
    preds_ptbdb = model_ptbdb.predict(beats, verbose=0)

    final_pred_mitbih = np.mean(preds_mitbih, axis=0, keepdims=True)
    final_pred_ptbdb = np.mean(preds_ptbdb, axis=0, keepdims=True)

    mitbih_probs = final_pred_mitbih.squeeze()
    mitbih_idx = np.argmax(mitbih_probs)
    mitbih_class = mitbih_labels[mitbih_idx]
    mitbih_confidence = mitbih_probs[mitbih_idx] * 100

    ptbdb_probs = final_pred_ptbdb.squeeze()
    ptbdb_idx = np.argmax(ptbdb_probs)
    ptbdb_class = ptbdb_labels[ptbdb_idx]
    ptbdb_confidence = ptbdb_probs[ptbdb_idx] * 100

    print("\n=== Combined ECG Prediction Summary ===\n")
    print(f"🧠 Beat morphology analysis:\n- Classified as '{mitbih_class}' with {mitbih_confidence:.1f}% confidence.\n")
    print(f"🫀 Heart disease screening:\n- Indicates '{ptbdb_class}' condition with {ptbdb_confidence:.1f}% confidence.\n")

    print("📋 Interpretation:")
    if mitbih_class.lower() == 'normal' and ptbdb_class.lower() == 'normal':
        print("- Beat morphology and disease screening both suggest normal findings.")
        print("- No immediate concerns detected.\n")
    else:
        print("- Note: Beat morphology refers to the shape of ECG signals.")
        print("- Disease status is based on broader patterns beyond individual beats.")
        print("- A normal beat does not rule out underlying heart conditions.")
        print("- Clinical review is recommended for further evaluation.\n")
    
    return beats.shape[0]
