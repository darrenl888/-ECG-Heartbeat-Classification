# ecg_processing_fast_modified.py (Upgraded Final Full Pipeline Version with Refined Peak Detection)

import numpy as np
from scipy.signal import butter, filtfilt, resample, find_peaks
import neurokit2 as nk
import json

###############################################################################
# Parameters
###############################################################################
GLOBAL_FS = 125
BEAT_WINDOW = 187
PEAK_DISTANCE = 30
PEAK_HEIGHT = 0.3
PEAK_PROMINENCE = 0.15
PEAKS_POLARITY_ABS = False
TACHY_THRESHOLD_BPM = 100

###############################################################################
# Quick validation
###############################################################################
def quick_validate_signal(signal, min_variance=1e-4):
    signal = np.asarray(signal)
    if np.any(np.isnan(signal)):
        return False
    if np.var(signal) < min_variance:
        return False
    if np.max(np.abs(signal)) > 5.0:
        return False
    return True

###############################################################################
# Utility - Refined Peak detection in beat snippet
###############################################################################
def _find_r_peaks_snippet(sig):
    work_sig = np.abs(sig) if PEAKS_POLARITY_ABS else sig
    raw_peaks, properties = find_peaks(
        work_sig,
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
        prominence=PEAK_PROMINENCE
    )

    if len(raw_peaks) == 0:
        return np.array([])

    # Refine peaks based on signal statistics
    strong_peaks = []
    mean_val = np.mean(work_sig)
    std_val = np.std(work_sig)
    threshold_dynamic = mean_val + 1.0 * std_val

    for peak_idx in raw_peaks:
        peak_val = work_sig[peak_idx]
        if peak_val >= 0.5 and peak_val >= threshold_dynamic:
            strong_peaks.append(peak_idx)

    return np.array(strong_peaks)

###############################################################################
# Global R-peak detection
###############################################################################
def detect_r_peaks(ecg_signal, fs=GLOBAL_FS):
    _, info = nk.ecg_process(ecg_signal, sampling_rate=fs, method="neurokit")
    return info["ECG_R_Peaks"]

###############################################################################
# Extract window around R-peak
###############################################################################
def _extract_window(sig, center, size):
    half = size // 2
    start, end = center - half, center + half
    if start < 0:
        return np.pad(sig[0:end], (abs(start), 0))
    if end > len(sig):
        return np.pad(sig[start:], (0, end - len(sig)))
    return sig[start:end]

###############################################################################
# Extract heartbeats
###############################################################################
def extract_heartbeat_windows(signal, r_peaks, mode='flag'):
    beats = []
    flags = {}
    for idx, r in enumerate(r_peaks):
        beat = _extract_window(signal, r, BEAT_WINDOW)
        
        # --- Fix beat length if necessary ---
        if beat.shape[0] < BEAT_WINDOW:
            beat = np.pad(beat, (0, BEAT_WINDOW - beat.shape[0]), mode='edge')
        elif beat.shape[0] > BEAT_WINDOW:
            beat = beat[:BEAT_WINDOW]

        n_peaks = len(_find_r_peaks_snippet(beat))

        if n_peaks == 1:
            beats.append(beat)
        else:
            if mode == 'flag':
                beats.append(beat)
                flags[len(beats)-1] = 'multi_peak'
            elif mode == 'drop':
                continue
            else:
                raise ValueError(f"Unknown mode: {mode}")
    return np.asarray(beats, dtype=np.float32), flags

###############################################################################
# Normalize beats
###############################################################################
def normalise_beats(beats):
    mins = beats.min(axis=1, keepdims=True)
    maxs = beats.max(axis=1, keepdims=True)
    rng  = np.where(maxs - mins == 0, 1e-8, maxs - mins)
    return (beats - mins) / rng

###############################################################################
# Preprocess raw ECG signal into beats
###############################################################################
def preprocess_raw_ecg_for_model(ecg_signal, original_fs, target_fs=GLOBAL_FS, apply_filter=True, tachy_mode='flag'):
    if not quick_validate_signal(ecg_signal):
        raise ValueError("Quick validation failed.")

    # --- Resample ---
    num_samples = int(len(ecg_signal) * target_fs / original_fs)
    signal = resample(ecg_signal, num_samples)

    # --- Apply Bandpass Filtering ---
    if apply_filter:
        lowcut = 0.5  # Hz
        highcut = 40.0  # Hz
        nyquist = 0.5 * target_fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # Create a bandpass Butterworth filter
        b_band, a_band = butter(3, [low, high], btype='band')  # Order 3 is good balance
        signal = filtfilt(b_band, a_band, signal)

    # --- R-peak detection ---
    r_peaks = detect_r_peaks(signal, target_fs)

    # --- Heartbeat extraction ---
    beats, flags = extract_heartbeat_windows(signal, r_peaks, mode=tachy_mode)

    # --- Normalize beats ---
    beats = normalise_beats(beats).astype(np.float32)

    return beats, flags

###############################################################################
# Analyze beats (BPM, tachycardia suspicion)
###############################################################################
def analyse_beats(beats, fs=GLOBAL_FS, tachy_threshold=TACHY_THRESHOLD_BPM):
    multi_idx, bpm_dict, tachy = [], {}, {}
    peak_counts = {}

    for i, beat in enumerate(beats):
        pk = _find_r_peaks_snippet(beat)
        peak_counts[i] = len(pk)

        if len(pk) > 1:
            multi_idx.append(i)
            rr = np.diff(pk)
            if rr.size:
                bpm = (fs * 60) / rr.mean()
                bpm_dict[i] = bpm
                tachy[i] = bpm > tachy_threshold

    return {
        "multi_indices": multi_idx,
        "bpm_dict": bpm_dict,
        "tachy_flags": tachy,
        "peak_counts": peak_counts
    }

###############################################################################
# Full Smart Preprocessing Pipeline without Model Prediction
###############################################################################
def smart_ecg_preprocessing(ecg_signal, original_fs, target_fs=GLOBAL_FS, apply_filter=True, tachy_mode='flag'):
    beats, flags = preprocess_raw_ecg_for_model(ecg_signal, original_fs, target_fs, apply_filter, tachy_mode)
    report = analyse_beats(beats, fs=target_fs)

    warnings = []

    for idx in report['multi_indices']:
        bpm = report['bpm_dict'].get(idx)
        if bpm is not None and report['tachy_flags'].get(idx, False):
            warnings.append((idx, bpm, 'tachycardia suspected'))
        else:
            warnings.append((idx, bpm, 'multiple peaks'))

    return {
        "beats": beats,
        "flags": flags,
        "report": report,
        "warnings": warnings
    }

###############################################################################
# New utility: Prepare arbitrary input (file / array / beat / recording) for model
###############################################################################
def prepare_input_for_model(input_data, original_fs=None):
    # --- Step 1: Load if file path ---
    if isinstance(input_data, str):
        if input_data.lower().endswith('.csv'):
            input_data = np.loadtxt(input_data, delimiter=',')
        elif input_data.lower().endswith('.json'):
            with open(input_data, 'r') as f:
                loaded = json.load(f)
            input_data = np.array(loaded)
        else:
            raise ValueError(f"Unsupported file format: {input_data}")

    # --- Step 2: Ensure numpy array ---
    input_data = np.asarray(input_data)

    # --- Step 3: Check if it's a full recording first ---
    if input_data.ndim == 1 and input_data.shape[0] > 2 * BEAT_WINDOW:
        if original_fs is None:
            raise ValueError("original_fs must be provided for full ECG recordings.")
        if not quick_validate_signal(input_data):
            raise ValueError("Input recording failed validation. May not be a valid ECG.")
        result = smart_ecg_preprocessing(input_data, original_fs)
        beats = result['beats']
        if beats.shape[0] == 0:
            raise ValueError("No valid beats extracted from the recording.")

    # --- Then check for a single beat ---
    elif input_data.ndim == 1:
        if not quick_validate_signal(input_data):
            raise ValueError("Input signal failed validation. May not be a valid ECG.")
        if input_data.shape[0] == BEAT_WINDOW:
            beats = input_data[np.newaxis, :]
        else:
            beats = resample(input_data, BEAT_WINDOW)[np.newaxis, :]

    # --- Batch of beats ---
    elif input_data.ndim == 2:
        if input_data.shape[1] == BEAT_WINDOW:
            beats = input_data
            for beat in beats:
                if not quick_validate_signal(beat):
                    raise ValueError("One or more beats failed validation. May not be valid ECG.")
        else:
            raise ValueError(f"Unexpected 2D input shape: {input_data.shape}. Each beat must be 187 samples.")
    else:
        raise ValueError(f"Unexpected input shape: {input_data.shape}")

    # --- Normalize if needed ---
    if np.max(beats) > 1.0 or np.min(beats) < 0.0:
        beats = normalise_beats(beats)

    # --- Add channel dimension ---
    beats = beats[..., np.newaxis]

    return beats
