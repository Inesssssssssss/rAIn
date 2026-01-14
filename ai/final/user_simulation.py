"""
Génération de profils utilisateurs et simulation de données physiologiques.
Basé sur les données réelles extraites par process_data.py
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict


# Charger les statistiques des données réelles
def load_real_statistics():
    """Charge les statistiques des données réelles pour calibrage."""
    try:
        df_real = pd.read_csv('features_dataset.csv')
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
    df_profiles.to_csv('user_profiles.csv', index=False)
    print(f"\n {len(profiles)} profils sauvegardés : user_profiles.csv")


