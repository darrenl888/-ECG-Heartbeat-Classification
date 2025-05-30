# 🫀 ECG Beat Morphology and Disease Screening

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ECG](https://img.shields.io/badge/Data-PhysioNet%20MITBIH%20%26%20PTBDB-critical)](https://physionet.org/)

---

This repository provides a clean end-to-end deep learning pipeline for:
- Preprocessing raw ECG recordings into standardized heartbeat windows
- Classifying each beat by **morphology** (e.g., normal, supraventricular, ventricular, etc.)
- Screening for underlying **heart disease** (normal vs abnormal)
- Generating smart, clinical-style interpretation summaries

Predictions are made using two separately trained CNN+LSTM models:
- `mitbih_model.h5` — trained on **MIT-BIH Arrhythmia Dataset** for 5-class beat classification
- `ptbdb_model.h5` — trained on **PTB Diagnostic ECG Dataset** for binary disease screening

---

## 📂 Files Overview

| File | Purpose |
|:---|:---|
| `ecg_processing_enhanced.py` | Full preprocessing pipeline: filtering, R-peak detection, heartbeat extraction, normalization, validation |
| `model_predictor.py` | Unified predictor using both models; generates clean clinical summaries |
| `Testing.ipynb` | Example usage, batch testing over PhysioNet records |
| `CNN+LSTM Models.ipynb` | Model training notebook using hybrid CNN+LSTM architectures |
| `models/mitbih_model.h5` | Trained model for beat morphology classification |
| `models/ptbdb_model.h5` | Trained model for disease screening |

---

## 🧠 Model Training (CNN + LSTM)

This repo includes code to train both classification models from scratch using a hybrid **CNN + LSTM** architecture:

| Model   | Dataset | Output Classes |
|---------|---------|----------------|
| MIT-BIH |5-class: Normal, SVEB, VEB, Fusion, Unknown | Normal, SVEB, VEB, Fusion, Unknown |
| PTBDB   |2-class: Normal, Abnormal | Normal, Abnormal |


Training highlights:
- Convolutional layers extract local ECG features
- LSTM layers capture sequential dependencies
- Trained using **focal loss**, early stopping, and class balancing
- Final models exported to `.h5` for seamless downstream inference

Use the `CNN+LSTM Models.ipynb` to retrain or modify architectures.

---

## ⚡ How It Works

1. **Preprocessing**  
   ECG signal is:
   - Bandpass filtered (0.5–40 Hz)
   - R-peaks detected using NeuroKit2
   - Centered 187-sample beats extracted
   - Noisy / multi-peak / tachycardic beats optionally flagged or dropped

2. **Prediction**  
   Each extracted beat is:
   - Fed into **two TensorFlow models** independently
   - Predictions are made **per beat**
   - Final output is the **mean prediction across all beats**

3. **Smart Reporting**  
   Combined summary includes:
   - Predicted beat type + confidence
   - Disease status + confidence
   - Explanation of possible contradictions
   - Clinical-style flag if abnormality is detected

---

## 🚀 Quickstart

### Install Requirements
```bash
pip install numpy scipy matplotlib tensorflow wfdb neurokit2
