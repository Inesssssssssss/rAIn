from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def _compute_basic_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Calcule un ensemble de features simples pour une fenêtre d'un canal.
    Features : mean, std, min, max, median, rms.
    """
    if signal.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "rms": np.nan,
        }

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    min_v = float(np.min(signal))
    max_v = float(np.max(signal))
    median = float(np.median(signal))
    rms = float(np.sqrt(np.mean(np.square(signal))))

    return {
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "median": median,
        "rms": rms,
    }


def records_to_features(
    records: List[Dict[str, Any]],
    window_seconds: int = 5,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Convertit une liste de "records" OpenSignals (dicts avec data/header)
    en DataFrame de features par fenêtres, prête pour du ML.

    Paramètres
    ----------
    records : List[Dict[str, Any]]
        Liste d'objets {"data": dict(channel->np.ndarray), "header": dict}.
    window_seconds : int
        Taille de fenêtre en secondes pour l'extraction de features.
    normalize : bool
        Si True, normalise les colonnes de features via z-score (par colonne).

    Retour
    ------
    df : pd.DataFrame
        DataFrame des features (une ligne par fenêtre et par fichier).
    scaler_params : Dict[str, Tuple[float, float]]
        Paramètres de normalisation (mean, std) par colonne, si normalize=True.
        Sinon, dict vide.
    """
    rows = []

    for file_idx, rec in enumerate(records):
        data = rec.get("data", {})
        header = rec.get("header", {})
        # Le sampling rate peut être dans header['sampling rate'] selon biosignalsnotebooks
        sr = header.get("sampling rate") or header.get("Sampling Rate")
        if sr is None:
            # fallback sécurisé
            raise ValueError("Le 'sampling rate' est introuvable dans le header.")

        # Détermine longueur commune minimale pour toutes les voies
        channels = list(data.keys())
        if not channels:
            continue

        min_len = min(len(np.asarray(data[ch])) for ch in channels)
        win_size = int(sr * window_seconds)
        if win_size <= 0:
            raise ValueError("'window_seconds' doit être > 0.")

        num_windows = max(0, min_len // win_size)

        for w in range(num_windows):
            start = w * win_size
            end = start + win_size

            row: Dict[str, Any] = {
                "file_index": file_idx,
                "window_index": w,
                "sampling_rate": sr,
            }

            for ch in channels:
                arr = np.asarray(data[ch])[start:end]
                feats = _compute_basic_features(arr)
                for k, v in feats.items():
                    row[f"{ch}_{k}"] = v

            rows.append(row)

    df = pd.DataFrame(rows)

    scaler_params: Dict[str, Tuple[float, float]] = {}
    if normalize and not df.empty:
        feature_cols = [c for c in df.columns if c not in ("file_index", "window_index", "sampling_rate")]
        for c in feature_cols:
            mean = float(df[c].mean())
            std = float(df[c].std())
            scaler_params[c] = (mean, std if std != 0 else 1.0)
            # z-score
            df[c] = (df[c] - mean) / (scaler_params[c][1])

    return df, scaler_params


def save_features_csv(df: pd.DataFrame, output_csv: str) -> None:
    """Enregistre la DataFrame de features en CSV."""
    df.to_csv(output_csv, index=False)
