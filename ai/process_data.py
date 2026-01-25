import biosignalsnotebooks as bsnb
import numpy as np
import pandas as pd
import neurokit2 as nk
import os
import re


def segment_signal(signal, fs, window_s=10, overlap=0.5):
    step = int(window_s * fs * (1 - overlap))
    size = int(window_s * fs)
    if step <= 0 or size <= 0:
        return []
    return [signal[i:i+size] for i in range(0, len(signal)-size+1, step)]


def ecg_features(ecg_segment, fs):
    try:
        _, info = nk.ecg_process(ecg_segment, sampling_rate=fs)
        hrv = nk.hrv_time(info, sampling_rate=fs, show=False)
        hr = nk.signal_rate(info.get("ECG_R_Peaks", np.array([])), sampling_rate=fs, desired_length=len(ecg_segment))
        return {
            "ecg_hr_bpm": float(np.nanmean(hr)) if len(hr) else np.nan,
            "ecg_hrv_sdnn": float(hrv.get("HRV_SDNN", np.nan)),
            "ecg_hrv_rmssd": float(hrv.get("HRV_RMSSD", np.nan)),
        }
    except Exception:
        return {"ecg_hr_bpm": np.nan, "ecg_hrv_sdnn": np.nan, "ecg_hrv_rmssd": np.nan}


def emg_features(emg_segment, fs):
    try:
        emg_cleaned = nk.emg_clean(emg_segment, sampling_rate=fs)
    except Exception:
        emg_cleaned = emg_segment
    emg_rect = np.abs(emg_cleaned)
    emg_rms = float(np.sqrt(np.mean(np.square(emg_rect)))) if emg_rect.size else np.nan
    nyquist = fs / 2.0
    low, high = 20.0, min(450.0, nyquist - 1.0)
    mf = np.nan
    if high > low:
        try:
            power = nk.signal_power(emg_cleaned, sampling_rate=fs, frequency_band=[low, high])
            mf = float(power.get("Median Frequency", np.nan))
        except Exception:
            pass
    return {"emg_rms": emg_rms, "emg_median_freq": mf}


def resp_features(resp_segment, fs):
    try:
        resp_cleaned = nk.signal_filter(resp_segment, sampling_rate=fs, lowcut=None, highcut=2, method="butterworth", order=4)
    except Exception:
        resp_cleaned = resp_segment
    try:
        rsp_signals, _ = nk.rsp_process(resp_cleaned, sampling_rate=fs)
        rate_series = rsp_signals.get("RSP_Rate")
        resp_rate = float(np.nanmean(rate_series)) if rate_series is not None else np.nan
    except Exception:
        resp_rate = np.nan
    return {"resp_rate_bpm": resp_rate}



def derive_resp_from_emg(emg_signal, fs):
    """Derive respiratory signal from EMG using filtering."""
    try:
        emg_cleaned = nk.emg_clean(emg_signal, sampling_rate=fs)
    except Exception:
        emg_cleaned = np.asarray(emg_signal)
    try:
        rect = np.abs(np.asarray(emg_cleaned))
        # enveloppe respiratoire très basse fréquence (lissage renforcé)
        env = nk.signal_filter(rect, sampling_rate=fs, lowcut=None, highcut=1.5, method="butterworth", order=4)
        return np.asarray(env)
    except Exception:
        return None


def basic_stats(x):
    """Compute basic statistics."""
    x = np.asarray(x)
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "rms": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
    }


def process_file(file_path, window_s=10, overlap=0.5, label=None, workout_id=None):
    """Load OpenSignals file, extract features by window."""
    data, header = bsnb.load(file_path, get_header=True)
    fs = int(header.get("sampling rate") or header.get("Sampling Rate"))

    # CH1=ECG, CH2=EMG (jambe), CH3=EMG respiratoire
    ecg = data.get('CH1')
    emg_leg = data.get('CH2')
    emg_resp = data.get('CH3')

    try:
        ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=fs, method="neurokit")
    except Exception:
        ecg_cleaned = ecg
    resp_from_emg = derive_resp_from_emg(np.asarray(emg_resp), fs) if emg_resp is not None else None
    ecg_windows = segment_signal(np.asarray(ecg_cleaned), fs, window_s, overlap) if ecg is not None else []
    emg_windows = segment_signal(np.asarray(emg_leg), fs, window_s, overlap) if emg_leg is not None else []
    resp_windows = segment_signal(np.asarray(resp_from_emg), fs, window_s, overlap) if resp_from_emg is not None else []

    n = min(len(ecg_windows) or 0, len(emg_windows) or 0, len(resp_windows) or 0)
    rows = []
    for i in range(n):
        row = {
            "file": os.path.basename(file_path),
            "window_index": i,
            "fs": fs,
        }
        row.update({f"ecg_{k}": v for k, v in basic_stats(ecg_windows[i]).items()})
        row.update(ecg_features(ecg_windows[i], fs))
        row.update({f"emg_{k}": v for k, v in basic_stats(emg_windows[i]).items()})
        row.update(emg_features(emg_windows[i], fs))
        row.update({f"resp_{k}": v for k, v in basic_stats(resp_windows[i]).items()})
        row.update(resp_features(resp_windows[i], fs))
        if label is not None:
            row["label"] = label
        if workout_id is not None:
            row["workout_id"] = workout_id
        rows.append(row)

    return pd.DataFrame(rows)


def build_dataset_from_folder(data_folder, window_s=20, overlap=0.5, label_map=None):
    """Build feature dataset from folder of OpenSignals files."""
    files = [f for f in os.listdir(data_folder) if f.lower().endswith('.txt')]
    dfs = []
    for name in files:
        fpath = os.path.join(data_folder, name)
        # Déduire activité depuis le nom
        activity = None
        lname = name.lower()
        if 'start' in lname:
            activity = 'start'
        elif 'sport' in lname:
            activity = 'sport'
        elif 'recovery' in lname:
            activity = 'recovery'
        # Extraire user id
        m = re.search(r'user(\d+)', lname)
        user_id = int(m.group(1)) if m else None
        
        # workout_id = user_id car pour un utilisateur donné, start/sport/recovery
        # correspondent à UNE SEULE séance d'entraînement continue
        workout_id = user_id if user_id is not None else None

        df = process_file(fpath, window_s=window_s, overlap=overlap, label=None, workout_id=workout_id)
        if df.empty:
            continue
        df['activity'] = activity
        if user_id is not None:
            df['user'] = user_id
        if label_map and activity in label_map:
            df['label'] = label_map[activity]
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "..", "..", "data")
    data_folder = os.path.normpath(data_folder)
    
    labels = {"start": 0, "sport": 1, "recovery": 0}
    df_all = build_dataset_from_folder(data_folder, window_s=20, overlap=0.5, label_map=labels)
    if not df_all.empty:
        out_csv = os.path.join(os.path.dirname(__file__), "features_dataset.csv")
        df_all.to_csv(out_csv, index=False)
        print(f"Dataset enrichi sauvegardé: {out_csv} ({len(df_all)} lignes)")
    else:
        print("Aucune donnée traitée dans le dossier data.")


