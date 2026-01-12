def generate_training_plan(age, level, goal):
    """
    Version minimale et explicable pour une démo académique.
    """

    base_plan = {
        "Débutant": 3,
        "Intermédiaire": 4,
        "Avancé": 5
    }

    sessions_per_week = base_plan.get(level, 3)

    plan = f"""
Profil utilisateur
------------------
Âge : {age}
Niveau : {level}
Objectif : {goal}

Plan d'entraînement (IA simulée)
--------------------------------
- {sessions_per_week} séances par semaine
- 1 sortie endurance fondamentale
- 1 séance fractionnée
- 1 sortie longue

Adaptation IA :
---------------
- Intensité ajustée selon le niveau
- Progression hebdomadaire simulée
- Réduction de charge si fatigue détectée
"""

    return plan.strip()


# -----------------------------------------------------------------------------
# Chargement simple OpenSignals TXT (.txt) depuis un dossier
# -----------------------------------------------------------------------------
from typing import Any, Dict, List
import os
import biosignalsnotebooks as bsnb


def load_txt_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load txt files from openSignals in a given folder.
    """

    if not os.path.isdir(folder_path):
        raise ValueError(f"Dossier introuvable: {folder_path}")

    results = []

    for name in os.listdir(folder_path):
        if not name.lower().endswith(".txt"):
            continue
        fpath = os.path.join(folder_path, name)
        if not os.path.isfile(fpath):
            continue

        data, header = bsnb.load(fpath, get_header=True)
        results.append({"data": data, "header": header})

    return results

# -----------------------------------------------------------------------------
# Prétraitement haut-niveau: charger un dossier et exporter des features en CSV
# -----------------------------------------------------------------------------
def preprocess_folder_to_csv(
    folder_path: str,
    output_csv: str,
    window_seconds: int = 5,
    normalize: bool = True
):
    """
    Charge les fichiers OpenSignals (.txt) d'un dossier, extrait des features
    par fenêtres et exporte le tout en CSV utilisable pour du Machine Learning.

    Paramètres
    ----------
    folder_path : str
        Chemin du dossier contenant les fichiers .txt OpenSignals.
    output_csv : str
        Chemin du fichier CSV à écrire.
    window_seconds : int
        Taille des fenêtres (en secondes) pour l'extraction de features.
    normalize : bool
        Si True, applique une normalisation (z-score) sur les features.
    """
    records = load_txt_files(folder_path)

    # Import local pour éviter les dépendances circulaires
    from .data_processing import records_to_features, save_features_csv

    df_features, _scaler_params = records_to_features(
        records,
        window_seconds=window_seconds,
        normalize=normalize,
    )

    save_features_csv(df_features, output_csv)


# -----------------------------------------------------------------------------
# Variante Python-only: retourner directement la DataFrame de features
# -----------------------------------------------------------------------------
def preprocess_folder(
    folder_path: str,
    window_seconds: int = 5,
    normalize: bool = True,
):
    """
    Charge les .txt OpenSignals dans `folder_path` et retourne
    directement la DataFrame de features prête pour du ML.

    Exemple:
        from ai.training_engine import preprocess_folder
        df = preprocess_folder(r"C:/path/vers/dossier")
    """
    records = load_txt_files(folder_path)

    # Import local pour éviter dépendances inutiles à l'import
    from data_processing import records_to_features

    df_features, _scaler_params = records_to_features(
        records,
        window_seconds=window_seconds,
        normalize=normalize,
    )
    return df_features

#f = preprocess_folder(r"C:/Users/inesr/Documents/OpenSignals (r)evolution/files/ai_adapt")
#print(f)