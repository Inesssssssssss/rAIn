"""
Générateur de données synthétiques pour augmenter le dataset d'entraînement.

BUT: Créer des features physiologiques réalistes pour entraîner le classificateur de fatigue.
Au lieu de générer des profils utilisateurs statiques, on génère directement
des features_dataset augmentées avec variation par âge/sexe/condition physique.

Les données sont générées autour des statistiques réelles observées,
avec variation physiologiquement plausible selon l'activité et le niveau de fatigue.
"""
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple


# Obtenir le répertoire du script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SyntheticFeatureGenerator:
    """Génère des features physiologiques synthétiques réalistes."""
    
    # Ranges réalistes pour les features (basées sur des données médicales)
    FEATURE_RANGES = {
        'ecg_hr_bpm': {'rest': (50, 70), 'light': (70, 100), 'moderate': (100, 140), 'intense': (140, 180)},
        'ecg_hrv_sdnn': {'low': (30, 100), 'normal': (100, 300), 'high': (300, 500)},
        'ecg_hrv_rmssd': {'low': (20, 50), 'normal': (50, 150), 'high': (150, 300)},
        'emg_rms': {'low': (0.5, 2.0), 'medium': (2.0, 8.0), 'high': (8.0, 25.0)},
        'emg_median_freq': {'low': (20, 100), 'medium': (100, 200), 'high': (200, 350)},
        'resp_rate_bpm': {'rest': (12, 16), 'light': (16, 20), 'moderate': (20, 25), 'intense': (25, 35)},
    }
    
    def __init__(self, random_state: int = 42):
        """Initialise le générateur."""
        np.random.seed(random_state)
    
    def _adjust_for_demographics(self, base_value: float, age: int, sex: str, 
                                 feature_name: str) -> float:
        """
        Ajuste une valeur de feature selon l'âge et le sexe.
        
        Args:
            base_value: Valeur de base
            age: Âge (18-70)
            sex: 'M' ou 'F'
            feature_name: Nom de la feature
            
        Returns:
            Valeur ajustée
        """
        # Effet de l'âge (généralement les vieux ont des HR plus élevées au repos)
        age_factor = 1.0
        if 'hr_bpm' in feature_name:
            age_factor = 1.0 + (age - 40) * 0.005  # +0.5% par 10 ans au-dessus de 40
        elif 'hrv' in feature_name:
            age_factor = 1.0 - (age - 40) * 0.01  # -1% par 10 ans au-dessus de 40
        
        # Effet du sexe (femmes ont généralement HR plus élevée, EMG parfois moins)
        sex_factor = 1.0
        if sex == 'F':
            if 'hr_bpm' in feature_name:
                sex_factor = 1.05  # +5%
            elif 'emg' in feature_name:
                sex_factor = 0.9  # -10%
        
        return base_value * age_factor * sex_factor
    
    def generate_window(self, age: int, sex: str, activity: str, fatigue_level: float,
                       fitness_level: str = 'intermediate') -> Dict[str, float]:
        """
        Génère une fenêtre (20s) de features physiologiques réalistes.
        
        Args:
            age: Âge (18-70)
            sex: 'M' ou 'F'
            activity: 'start' (échauffement), 'sport' (effort), 'recovery' (récupération)
            fatigue_level: 0.0-1.0 (0=pas fatigué, 1=très fatigué)
            fitness_level: 'beginner', 'intermediate', 'advanced'
            
        Returns:
            Dict avec toutes les features
        """
        # Intensité selon l'activité
        intensity_map = {'start': 'light', 'sport': 'intense', 'recovery': 'moderate'}
        base_intensity = intensity_map.get(activity, 'moderate')
        
        # Ajuster l'intensité selon le fitness (les athlètes ont des HR plus basses au même effort)
        fitness_adjustment = 1.0 if fitness_level == 'intermediate' else (0.85 if fitness_level == 'advanced' else 1.15)
        
        # HR dépend de : intensité + fatigue (la fatigue augmente la HR)
        hr_min, hr_max = self.FEATURE_RANGES['ecg_hr_bpm'][base_intensity]
        hr = np.random.normal((hr_min + hr_max) / 2, (hr_max - hr_min) / 6)
        hr = hr * fitness_adjustment + fatigue_level * 15  # La fatigue augmente HR
        hr = np.clip(hr, 40, 200)
        hr = self._adjust_for_demographics(hr, age, sex, 'ecg_hr_bpm')
        
        # HRV : dépend de la fatigue (la fatigue réduit HRV)
        # Au repos + fatigué : HRV très basse. En effort intense : HRV naturellement basse.
        if fatigue_level > 0.7 or (activity == 'sport' and fatigue_level > 0.3):
            hrv_level = 'low'
        elif fatigue_level > 0.3:
            hrv_level = 'normal'
        else:
            hrv_level = 'high'
        
        hrv_sdnn_min, hrv_sdnn_max = self.FEATURE_RANGES['ecg_hrv_sdnn'][hrv_level]
        ecg_hrv_sdnn = np.random.normal((hrv_sdnn_min + hrv_sdnn_max) / 2, (hrv_sdnn_max - hrv_sdnn_min) / 6)
        ecg_hrv_sdnn = np.clip(ecg_hrv_sdnn, 20, 600)
        ecg_hrv_sdnn = self._adjust_for_demographics(ecg_hrv_sdnn, age, sex, 'ecg_hrv_sdnn')
        
        hrv_rmssd_min, hrv_rmssd_max = self.FEATURE_RANGES['ecg_hrv_rmssd'][hrv_level]
        ecg_hrv_rmssd = np.random.normal((hrv_rmssd_min + hrv_rmssd_max) / 2, (hrv_rmssd_max - hrv_rmssd_min) / 6)
        ecg_hrv_rmssd = np.clip(ecg_hrv_rmssd, 10, 400)
        
        # EMG : dépend de l'intensité et de la fatigue
        if activity == 'sport' or fatigue_level > 0.5:
            emg_level = 'high' if fatigue_level > 0.6 else 'medium'
        else:
            emg_level = 'low' if fatigue_level < 0.3 else 'medium'
        
        emg_rms_min, emg_rms_max = self.FEATURE_RANGES['emg_rms'][emg_level]
        emg_rms = np.random.normal((emg_rms_min + emg_rms_max) / 2, (emg_rms_max - emg_rms_min) / 6)
        emg_rms = np.clip(emg_rms, 0.1, 30)
        emg_rms = self._adjust_for_demographics(emg_rms, age, sex, 'emg_rms')
        
        emg_freq_min, emg_freq_max = self.FEATURE_RANGES['emg_median_freq'][emg_level]
        emg_median_freq = np.random.normal((emg_freq_min + emg_freq_max) / 2, (emg_freq_max - emg_freq_min) / 6)
        emg_median_freq = np.clip(emg_median_freq, 10, 400)
        
        # Respiration : dépend de l'intensité et de la fatigue
        resp_intensity = base_intensity if fatigue_level < 0.4 else 'moderate'
        resp_min, resp_max = self.FEATURE_RANGES['resp_rate_bpm'][resp_intensity]
        resp_rate = np.random.normal((resp_min + resp_max) / 2, (resp_max - resp_min) / 6)
        resp_rate = resp_rate + fatigue_level * 5  # La fatigue augmente la respiration
        resp_rate = np.clip(resp_rate, 10, 40)
        
        # Calculer les statistiques ECG, EMG, Resp (moyennes sur la fenêtre)
        # Générées directement pour la fenêtre
        ecg_signal = np.random.normal(0, 100, 200)  # 200 samples @ 1000 Hz = 0.2s, représente fenêtre 10s
        emg_signal = np.random.normal(0, 5, 200)
        resp_signal = np.random.normal(0, 1, 200)
        
        return {
            'ecg_mean': float(np.mean(ecg_signal)),
            'ecg_std': float(np.std(ecg_signal)),
            'ecg_min': float(np.min(ecg_signal)),
            'ecg_max': float(np.max(ecg_signal)),
            'ecg_median': float(np.median(ecg_signal)),
            'ecg_rms': float(np.sqrt(np.mean(np.square(ecg_signal)))),
            'ecg_hr_bpm': float(hr),
            'ecg_hrv_sdnn': float(ecg_hrv_sdnn),
            'ecg_hrv_rmssd': float(ecg_hrv_rmssd),
            'emg_mean': float(np.mean(np.abs(emg_signal))),
            'emg_std': float(np.std(emg_signal)),
            'emg_min': float(0.0),
            'emg_max': float(np.max(np.abs(emg_signal))),
            'emg_median': float(np.median(np.abs(emg_signal))),
            'emg_rms': float(emg_rms),
            'emg_median_freq': float(emg_median_freq),
            'resp_mean': float(np.mean(resp_signal)),
            'resp_std': float(np.std(resp_signal)),
            'resp_min': float(np.min(resp_signal)),
            'resp_max': float(np.max(resp_signal)),
            'resp_median': float(np.median(resp_signal)),
            'resp_rms': float(np.sqrt(np.mean(np.square(resp_signal)))),
            'resp_rate_bpm': float(resp_rate),
        }
























    def generate_session(self, user_id: int, age: int, sex: str, 
                        fitness_level: str = 'intermediate',
                        n_windows: int = 48) -> pd.DataFrame:
        """
        Génère une séance complète (start + sport + recovery).
        
        Args:
            user_id: ID utilisateur
            age: Âge
            sex: Sexe ('M' ou 'F')
            fitness_level: 'beginner', 'intermediate', 'advanced'
            n_windows: Nombre total de fenêtres (divisé entre les 3 phases)
            
        Returns:
            DataFrame avec tous les samples de la séance
        """
        windows_per_phase = n_windows // 3
        rows = []
        window_idx = 0

        # PHASE 1 : START (échauffement, pas fatigué)
        for i in range(windows_per_phase):
            # Fatigue augmente légèrement durant l'échauffement (0.0 → 0.1)
            fatigue_level = 0.0 + i / (windows_per_phase * 20)
            features = self.generate_window(age, sex, 'start', fatigue_level, fitness_level)
            features['activity'] = 'start'
            features['user'] = user_id
            features['window_index'] = window_idx
            features['fs'] = 1000
            features['file'] = f'start_user{user_id}.txt'
            features['label'] = 0  # Pas fatigué
            rows.append(features)
            window_idx += 1
        
        # PHASE 2 : SPORT (effort intensif, accumulation de fatigue)
        for i in range(windows_per_phase):
            # Fatigue augmente fortement durant le sport (0.1 → 0.8)
            fatigue_level = 0.1 + (i / windows_per_phase) * 0.7
            features = self.generate_window(age, sex, 'sport', fatigue_level, fitness_level)
            features['activity'] = 'sport'
            features['user'] = user_id
            features['window_index'] = window_idx
            features['fs'] = 1000
            features['file'] = f'sport_user{user_id}.txt'
            # Label: fatigué si la seconde moitié du sport
            features['label'] = 1 if i > windows_per_phase / 2 else 0
            rows.append(features)
            window_idx += 1
        
        # PHASE 3 : RECOVERY (récupération, fatigue décroît)
        for i in range(windows_per_phase):
            # Fatigue diminue durant la récupération (0.8 → 0.3)
            fatigue_level = 0.8 - (i / windows_per_phase) * 0.5
            features = self.generate_window(age, sex, 'recovery', fatigue_level, fitness_level)
            features['activity'] = 'recovery'
            features['user'] = user_id
            features['window_index'] = window_idx
            features['fs'] = 1000
            features['file'] = f'recovery_user{user_id}.txt'
            # Label: fatigué dans la première moitié de la récupération
            features['label'] = 1 if i < windows_per_phase / 2 else 0
            rows.append(features)
            window_idx += 1
        
        return pd.DataFrame(rows)


def generate_synthetic_dataset(n_users: int = 100, 
                              sessions_per_user: int = 3,
                              windows_per_session: int = 48,
                              output_path: str = None) -> pd.DataFrame:

    """
    Génère un dataset synthétique complet.








    
    Args:
        n_users: Nombre d'utilisateurs synthétiques
        sessions_per_user: Nombre de séances par utilisateur
        windows_per_session: Nombre de fenêtres par séance
        output_path: Chemin pour sauvegarder (si None, retourne DataFrame)

        
    Returns:
        DataFrame combiné
    """
    generator = SyntheticFeatureGenerator()
    all_rows = []

    print(f"Génération de {n_users} utilisateurs × {sessions_per_user} séances...")
    print(f"Total: {n_users * sessions_per_user * windows_per_session} fenêtres")


    for user_id in range(n_users):
        # Variation démographique
        age = np.random.randint(18, 71)
        sex = np.random.choice(['M', 'F'], p=[0.5, 0.5])
        fitness = np.random.choice(['beginner', 'intermediate', 'advanced'], p=[0.33, 0.34, 0.33])
        
        for session in range(sessions_per_user):
            df_session = generator.generate_session(
                user_id, age, sex, fitness, windows_per_session
            )
            all_rows.append(df_session)
            
            if (user_id + 1) % 10 == 0 and session == sessions_per_user - 1:
                print(f"  ✓ {user_id + 1} utilisateurs générés...")

    df = pd.concat(all_rows, ignore_index=True)



    # Réorganiser les colonnes
    cols = ['file', 'window_index', 'fs', 'ecg_mean', 'ecg_std', 'ecg_min', 'ecg_max',
            'ecg_median', 'ecg_rms', 'ecg_hr_bpm', 'ecg_hrv_sdnn', 'ecg_hrv_rmssd',
            'emg_mean', 'emg_std', 'emg_min', 'emg_max', 'emg_median', 'emg_rms',
            'emg_median_freq', 'resp_mean', 'resp_std', 'resp_min', 'resp_max',
            'resp_median', 'resp_rms', 'resp_rate_bpm', 'activity', 'user', 'label']
    df = df[cols]

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n Dataset sauvegardé: {output_path}")
        print(f"  Total: {len(df)} samples")
        print(f"  Fatigués: {(df['label']==1).sum()} ({100*(df['label']==1).sum()/len(df):.1f}%)")
        print(f"  Utilisateurs: {df['user'].nunique()}")
        print(f"  Activités: {df['activity'].unique().tolist()}")

    return df


def augment_existing_dataset(features_csv: str, 
                            n_synthetic_users: int = 50,
                            output_path: str = None) -> pd.DataFrame:
    """
    Augmente un dataset existant avec des données synthétiques.
    
    Args:
        features_csv: Chemin vers features_dataset.csv existant
        n_synthetic_users: Nombre d'utilisateurs synthétiques à ajouter
        output_path: Chemin pour sauvegarder
        
    Returns:
        DataFrame combiné
    """
    # Charger dataset réel
    df_real = pd.read_csv(features_csv)
    n_real_users = df_real['user'].max() + 1

    print(f"Dataset réel chargé: {len(df_real)} samples, {n_real_users} utilisateurs")


    # Générer dataset synthétique
    df_synthetic = generate_synthetic_dataset(
        n_users=n_synthetic_users,
        sessions_per_user=2,
        windows_per_session=48,
        output_path=None
    )

    # Décaler les user_id pour éviter les collisions
    df_synthetic['user'] = df_synthetic['user'] + n_real_users


    # Combiner
    df_combined = pd.concat([df_real, df_synthetic], ignore_index=True)






    if output_path:
        df_combined.to_csv(output_path, index=False)
        print(f"\n✓ Dataset augmenté sauvegardé: {output_path}")
        print(f"  Données réelles: {len(df_real)} samples ({n_real_users} users)")
        print(f"  Données synthétiques: {len(df_synthetic)} samples ({n_synthetic_users} users)")
        print(f"  Total: {len(df_combined)} samples ({df_combined['user'].nunique()} users)")
        print(f"  Fatigués: {(df_combined['label']==1).sum()} ({100*(df_combined['label']==1).sum()/len(df_combined):.1f}%)")

    return df_combined



if __name__ == "__main__":
    import sys
    
    features_path = os.path.join(SCRIPT_DIR, 'features_dataset.csv')
    
    if os.path.exists(features_path):
        output_path = os.path.join(SCRIPT_DIR, 'features_dataset_augmented.csv')
        df = augment_existing_dataset(
            features_csv=features_path,
            n_synthetic_users=100,  # Ajouter 100 utilisateurs synthétiques
            output_path=output_path
        )
        
        print("\n=== STATISTIQUES FINALES ===")
        print(f"Total samples: {len(df)}")
        for activity in sorted(df['activity'].unique()):
            mask = df['activity'] == activity
            n_fatigued = (df.loc[mask, 'label'] == 1).sum()
            n_total = mask.sum()
            print(f"{activity:12s}: {n_total:5d} samples ({n_fatigued:5d} fatigués, {100*n_fatigued/n_total:5.1f}%)")
        
    else:
        print(f"Fichier non trouvé: {features_path}")