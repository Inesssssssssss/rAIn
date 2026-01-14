"""
Classificateur Random Forest pour la prédiction de la fatigue.
Utilise les features physiologiques: ECG, EMG et respiration.
"""
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)
import joblib
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Obtenir le répertoire du script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FatigueClassifier:
    """
    Classificateur Random Forest pour la prédiction de la fatigue.
    
    Features utilisées:
    - ECG: ecg_hr_bpm, ecg_hrv_sdnn, ecg_hrv_rmssd
    - EMG jambe: emg_rms, emg_median_freq
    - Respiration: resp_rate_bpm
    """
    
    FEATURE_NAMES = [
        'ecg_hr_bpm', 
        'ecg_hrv_sdnn', 
        'ecg_hrv_rmssd',
        'emg_rms', 
        'emg_median_freq',
        'resp_rate_bpm'
    ]
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialise le classificateur.
        
        Args:
            n_estimators: Nombre d'arbres dans la forêt
            random_state: Pour reproductibilité
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance_ = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df: DataFrame contenant les features et une colonne 'fatigue' (0 ou 1)
            
        Returns:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        """
        # Vérifier que toutes les features existent
        missing_features = set(self.FEATURE_NAMES) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features manquantes dans les données: {missing_features}")
        
        if 'fatigue' not in df.columns:
            raise ValueError("La colonne 'fatigue' est requise dans les données")
        
        # Extraire features et labels
        X = df[self.FEATURE_NAMES].values
        y = df['fatigue'].values
        
        # Gérer les valeurs manquantes
        # Remplacer les NaN par la médiane de la colonne
        X_clean = X.copy()
        for i in range(X.shape[1]):
            col_data = X[:, i]
            mask = ~np.isnan(col_data)
            if mask.sum() > 0:
                median_val = np.nanmedian(col_data)
                X_clean[np.isnan(col_data), i] = median_val
            else:
                X_clean[:, i] = 0  # Si tout est NaN, remplir avec 0
        
        return X_clean, y
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, validation: bool = True) -> Dict:
        """
        Entraîne le modèle.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion de test
            validation: Si True, affiche les métriques de validation
            
        Returns:
            Dict contenant les métriques d'entraînement et test
        """
        # Normaliser les features
        X_scaled = self.scaler.fit_transform(X)
        
        # Diviser train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Entraîner
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Sauvegarder l'importance des features
        self.feature_importance_ = pd.DataFrame({
            'feature': self.FEATURE_NAMES,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Prédictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Métriques
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred, zero_division=0)
        }
        
        if validation:
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='f1')
            metrics['cv_f1_scores'] = cv_scores
            metrics['cv_f1_mean'] = cv_scores.mean()
            metrics['cv_f1_std'] = cv_scores.std()
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la fatigue sur de nouvelles données.
        
        Args:
            X: Features (doit avoir les mêmes colonnes que FEATURE_NAMES)
            
        Returns:
            Prédictions (0: non fatigué, 1: fatigué)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la probabilité de fatigue sur de nouvelles données.
        
        Args:
            X: Features
            
        Returns:
            Probabilités pour chaque classe
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features."""
        if self.feature_importance_ is None:
            raise ValueError("Le modèle doit être entraîné d'abord")
        return self.feature_importance_
    
    def save(self, model_path: str, scaler_path: str):
        """Sauvegarde le modèle et le scaler."""
        # Utiliser des chemins absolus si relatifs
        if not os.path.isabs(model_path):
            model_path = os.path.join(SCRIPT_DIR, model_path)
        if not os.path.isabs(scaler_path):
            scaler_path = os.path.join(SCRIPT_DIR, scaler_path)
            
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Modèle sauvegardé: {model_path}")
        print(f"Scaler sauvegardé: {scaler_path}")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str):
        """Charge un modèle entraîné."""
        # Utiliser des chemins absolus si relatifs
        if not os.path.isabs(model_path):
            model_path = os.path.join(SCRIPT_DIR, model_path)
        if not os.path.isabs(scaler_path):
            scaler_path = os.path.join(SCRIPT_DIR, scaler_path)
            
        clf = cls()
        clf.model = joblib.load(model_path)
        clf.scaler = joblib.load(scaler_path)
        clf.is_fitted = True
        return clf


def train_fatigue_model_from_simulations(
    simulations_path: str,
    output_model_path: str = 'fatigue_model.pkl',
    output_scaler_path: str = 'fatigue_scaler.pkl'
) -> FatigueClassifier:
    """
    Entraîne le modèle de fatigue à partir de données simulées.
    
    Args:
        simulations_path: Chemin vers le CSV contenant les simulations
        output_model_path: Chemin pour sauvegarder le modèle
        output_scaler_path: Chemin pour sauvegarder le scaler
        
    Returns:
        Modèle entraîné
    """
    # Utiliser le chemin absolu si relatif
    if not os.path.isabs(simulations_path):
        simulations_path = os.path.join(SCRIPT_DIR, simulations_path)
        
    print(f"Chargement des données de {simulations_path}...")
    df = pd.read_csv(simulations_path)
    
    print(f"Données chargées: {df.shape[0]} samples")
    print(f"Colonnes: {list(df.columns)}")
    
    # Initialiser et entraîner le classificateur
    clf = FatigueClassifier()
    X, y = clf.prepare_data(df)
    
    print(f"\nEntraînement du modèle avec {X.shape[0]} samples...")
    print(f"Distribution des classes: Non fatigué={np.sum(y==0)}, Fatigué={np.sum(y==1)}")
    
    metrics = clf.train(X, y, validation=True)
    
    # Afficher les résultats
    print("\n=== RÉSULTATS D'ENTRAÎNEMENT ===")
    print(f"Accuracy (train): {metrics['train_accuracy']:.3f}")
    print(f"Accuracy (test):  {metrics['test_accuracy']:.3f}")
    print(f"Precision:        {metrics['test_precision']:.3f}")
    print(f"Recall:           {metrics['test_recall']:.3f}")
    print(f"F1-Score:         {metrics['test_f1']:.3f}")
    
    if 'cv_f1_mean' in metrics:
        print(f"\nCross-Validation F1: {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})")
    
    print("\n=== IMPORTANCE DES FEATURES ===")
    print(clf.get_feature_importance())
    
    print("\n=== RAPPORT DE CLASSIFICATION ===")
    print(metrics['classification_report'])
    
    print("\n=== MATRICE DE CONFUSION ===")
    print("[[TN, FP],")
    print(" [FN, TP]]")
    print(metrics['confusion_matrix'])
    
    # Sauvegarder le modèle
    clf.save(output_model_path, output_scaler_path)
    
    return clf


if __name__ == "__main__":
    # Exemple d'utilisation
    import sys
    
    simulations_file = 'features_dataset.csv'
    
    try:
        clf = train_fatigue_model_from_simulations(simulations_file)
        print("\n✓ Modèle entraîné avec succès!")
    except FileNotFoundError:
        features_path = os.path.join(SCRIPT_DIR, simulations_file)
        print(f"Erreur: Le fichier {features_path} n'a pas été trouvé.")
        print("Assurez-vous que features_dataset.csv existe dans le répertoire du script.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
