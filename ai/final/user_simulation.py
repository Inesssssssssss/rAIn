"""
Génération de profils utilisateurs et simulation de données physiologiques.
Basé sur les données réelles extraites par process_data.py
"""
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Obtenir le répertoire du script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Charger les statistiques des données réelles
def load_real_statistics():
    """Charge les statistiques des données réelles pour calibrage."""
    try:
        features_path = os.path.join(SCRIPT_DIR, 'features_dataset.csv')
        df_real = pd.read_csv(features_path)
        # Statistiques réelles observées
        stats = {
            'HR_mean': df_real['ecg_hr_bpm'].mean(),
            'HR_std': df_real['ecg_hr_bpm'].std(),
            'HR_min': df_real['ecg_hr_bpm'].min(),
            'HR_max': df_real['ecg_hr_bpm'].max(),
            'RR_mean': df_real['resp_rate_bpm'].mean(),
            'RR_std': df_real['resp_rate_bpm'].std(),
            'RR_min': df_real['resp_rate_bpm'].min(),
            'RR_max': df_real['resp_rate_bpm'].max(),
        }
        return stats
    except:
        # Valeurs par défaut si fichier non trouvé
        return {
            'HR_mean': 60, 'HR_std': 15, 'HR_min': 38, 'HR_max': 80,
            'RR_mean': 30, 'RR_std': 5, 'RR_min': 27, 'RR_max': 44
        }

REAL_STATS = load_real_statistics()


@dataclass
class UserProfile:
    """Profil physiologique d'une utilisatrice (calibré sur données réelles)."""
    age: int
    sex: str
    fitness_level: str
    HR_repos: float  # HR au repos observée dans données réelles
    HR_max: float   # HR max potentielle
    HR_reserve: float
    RR_repos: float  # RR au repos
    activity_type: str  # récupération, sport, etc.


def generate_user_profile(age=None, sex=None, fitness_level='intermédiaire') -> UserProfile:
    """
    Génère un profil utilisateur.
    - fitness_level='intermédiaire' : utilise les données réelles 
    - fitness_level='débutant' ou 'avancé' : génération aléatoire
    """
    
    if fitness_level == 'intermédiaire':
        age = np.random.randint(20, 50) if age is None else age
        sex = np.random.choice(['M', 'F']) if sex is None else sex
        
        # HR_repos basée sur données réelles
        HR_repos = np.clip(
            np.random.normal(REAL_STATS['HR_mean'], REAL_STATS['HR_std']/3),
            REAL_STATS['HR_min'], 
            REAL_STATS['HR_max']
        )
        
        # RR_repos basée sur données réelles
        RR_repos = np.clip(
            np.random.normal(REAL_STATS['RR_mean'], REAL_STATS['RR_std']/3),
            REAL_STATS['RR_min'],
            REAL_STATS['RR_max']
        )
    else:
        # Profils DÉBUTANT et AVANCÉ générés aléatoirement
        age = np.random.randint(20, 50) if age is None else age
        sex = np.random.choice(['M', 'F']) if sex is None else sex
        
        if fitness_level == 'débutant':
            HR_repos = np.random.normal(75, 8)  # HR plus élevée au repos
            RR_repos = np.random.normal(18, 3)  # RR plus élevée
        else:  # avancé
            HR_repos = np.random.normal(55, 5)  # HR plus basse au repos
            RR_repos = np.random.normal(12, 2)  # RR plus basse
        
        HR_repos = np.clip(HR_repos, 45, 90)
        RR_repos = np.clip(RR_repos, 10, 25)
    
    # HR_max selon formule Tanaka adaptée
    if sex == 'F':
        HR_max = 206 - 0.88 * age + np.random.normal(0, 5)
    else:
        HR_max = 208 - 0.7 * age + np.random.normal(0, 5)
    
    HR_reserve = HR_max - HR_repos
    
    return UserProfile(
        age=age,
        sex=sex,
        fitness_level=fitness_level,
        HR_repos=HR_repos,
        HR_max=HR_max,
        HR_reserve=HR_reserve,
        RR_repos=RR_repos,
        activity_type='mixed'
    )


def generate_fatigue_labels(df: pd.DataFrame,
                           hr_percentile: float = 75.0,
                           hrv_percentile: float = 25.0,
                           emg_percentile: float = 75.0,
                           resp_percentile: float = 75.0) -> pd.DataFrame:
    """
    Génère les labels de fatigue basés sur des percentiles des features physiologiques.
    
    Logique:
    - HR élevé (> 75e percentile) = fatigue
    - HRV basse (< 25e percentile) = fatigue
    - EMG élevé (> 75e percentile) = fatigue
    - Respiration élevée (> 75e percentile) = fatigue
    
    Un sample est "fatigué" si au moins 2 de ces conditions sont vraies.
    
    Args:
        df: DataFrame contenant les features physiologiques
        hr_percentile: Percentile pour HR élevé
        hrv_percentile: Percentile pour HRV basse
        emg_percentile: Percentile pour EMG élevé
        resp_percentile: Percentile pour respiration élevée
        
    Returns:
        DataFrame avec la colonne 'fatigue' ajoutée
    """
    required_cols = ['ecg_hr_bpm', 'ecg_hrv_sdnn', 'ecg_hrv_rmssd', 
                     'emg_rms', 'emg_median_freq', 'resp_rate_bpm']
    
    # Vérifier que les colonnes existent
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Features manquantes. Colonnes requises: {required_cols}")
    
    # Créer des copies pour éviter les warnings SettingWithCopyWarning
    df = df.copy()
    
    # 1. HR élevé (75e percentile)
    hr_threshold = df['ecg_hr_bpm'].quantile(hr_percentile / 100.0)
    high_hr = df['ecg_hr_bpm'] > hr_threshold
    
    # 2. HRV basse (25e percentile)
    hrv_sdnn_threshold = df['ecg_hrv_sdnn'].quantile(hrv_percentile / 100.0)
    low_hrv_sdnn = df['ecg_hrv_sdnn'] < hrv_sdnn_threshold
    
    hrv_rmssd_threshold = df['ecg_hrv_rmssd'].quantile(hrv_percentile / 100.0)
    low_hrv_rmssd = df['ecg_hrv_rmssd'] < hrv_rmssd_threshold
    
    low_hrv = low_hrv_sdnn | low_hrv_rmssd
    
    # 3. EMG élevé (75e percentile)
    emg_rms_threshold = df['emg_rms'].quantile(emg_percentile / 100.0)
    high_emg_rms = df['emg_rms'] > emg_rms_threshold
    
    emg_freq_threshold = df['emg_median_freq'].quantile(emg_percentile / 100.0)
    high_emg_freq = df['emg_median_freq'] > emg_freq_threshold
    
    high_emg = high_emg_rms | high_emg_freq
    
    # 4. Respiration élevée (75e percentile)
    resp_threshold = df['resp_rate_bpm'].quantile(resp_percentile / 100.0)
    high_resp = df['resp_rate_bpm'] > resp_threshold
    
    # Compter les indicateurs de fatigue par sample
    fatigue_indicators = (
        high_hr.fillna(False).astype(int) +
        low_hrv.fillna(False).astype(int) +
        high_emg.fillna(False).astype(int) +
        high_resp.fillna(False).astype(int)
    )
    
    # Un sample est fatigué si au moins 2 indicateurs sont positifs
    df['fatigue'] = (fatigue_indicators >= 2).astype(int)
    
    return df


if __name__ == "__main__":
    print("\n=== Génération de profils utilisateurs ===")
    
    profiles = []
    for i in range(50):
        fitness = np.random.choice(['débutant', 'intermédiaire', 'avancé'], p=[0.33, 0.34, 0.33])
        user = generate_user_profile(fitness_level=fitness)
        profiles.append({
            'user_id': i,
            'age': user.age,
            'sex': user.sex,
            'fitness': fitness,
            'HR_repos': round(user.HR_repos, 1),
            'HR_max': round(user.HR_max, 1),
            'RR_repos': round(user.RR_repos, 1)
        })
    
    df_profiles = pd.DataFrame(profiles)
    profiles_path = os.path.join(SCRIPT_DIR, 'user_profiles.csv')
    df_profiles.to_csv(profiles_path, index=False)
    print(f"\n {len(profiles)} profils sauvegardés : user_profiles.csv")
    
    # Charger les données simulées
    try:
        features_path = os.path.join(SCRIPT_DIR, 'features_dataset.csv')
        df_features = pd.read_csv(features_path)
        print(f"\n {len(df_features)} samples chargés : features_dataset.csv")
        
        # Générer les labels de fatigue
        df_features = generate_fatigue_labels(df_features)
        
        # Sauvegarder avec les labels
        df_features.to_csv(features_path, index=False)
        print(f"Labels de fatigue ajoutés et fichier sauvegardé!")
        
        # Afficher statistiques
        n_fatigued = (df_features['fatigue'] == 1).sum()
        n_not_fatigued = (df_features['fatigue'] == 0).sum()
        
        print(f"\n=== DISTRIBUTION DES LABELS DE FATIGUE ===")
        print(f"Non-fatigué (0): {n_not_fatigued} ({100*n_not_fatigued/len(df_features):.1f}%)")
        print(f"Fatigué (1):     {n_fatigued} ({100*n_fatigued/len(df_features):.1f}%)")
        
        if 'activity' in df_features.columns:
            print(f"\n=== DISTRIBUTION PAR ACTIVITÉ ===")
            for activity in sorted(df_features['activity'].unique()):
                mask = df_features['activity'] == activity
                fatigued = (df_features.loc[mask, 'fatigue'] == 1).sum()
                total = mask.sum()
                print(f"{activity:15s}: {fatigued:3d}/{total:3d} ({100*fatigued/total:5.1f}%)")
    
    except FileNotFoundError:
        print("Aucun fichier features_dataset.csv trouvé. Générez d'abord les données avec process_data.py")



