"""
Système de recommandation d'entraînement personnalisé.
Analyse l'historique de l'utilisateur et propose une séance adaptée.
"""
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List, Tuple

# Obtenir le répertoire du script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from ai.fatigue_classifier import FatigueClassifier

# Obtenir le répertoire du script (déjà défini)


class TrainingRecommender:
    """
    Système de recommandation d'entraînement personnalisé.
    """
    
    def __init__(self, model_path: str = None, 
                 scaler_path: str = None):
        """
        Initialise le recommandeur.
        
        Args:
            model_path: Chemin vers le modèle de fatigue entraîné
            scaler_path: Chemin vers le scaler
        """
        # Utiliser les chemins par défaut s'ils ne sont pas fournis
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'models', 'fatigue_model.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(PROJECT_ROOT, 'models', 'fatigue_scaler.pkl')
        
        self.classifier = FatigueClassifier.load(model_path, scaler_path)
        self.user_history = None
        self.user_profile = None
        
    def load_user_data(self, user_id: int, 
                       features_path: str = None,
                       profiles_path: str = None) -> bool:
        """
        Charge les données d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            features_path: Chemin vers les features (défaut: output/features/features_dataset.csv)
            profiles_path: Chemin vers les profils utilisateurs (défaut: output/user_profiles.csv)
            
        Returns:
            True si les données ont été chargées avec succès
        """
        # Définir les chemins par défaut basés sur la structure du projet
        if features_path is None:
            features_path = os.path.join(PROJECT_ROOT, 'output', 'features', 'features_dataset.csv')
        if profiles_path is None:
            profiles_path = os.path.join(PROJECT_ROOT, 'output', 'user_profiles.csv')
        
        # Essayer d'abord le dataset augmenté
        features_augmented_path = features_path.replace('features_dataset.csv', 'features_dataset_augmented.csv')
        if os.path.exists(features_augmented_path):
            features_path = features_augmented_path
        
        df_features = pd.read_csv(features_path)
        
        # Filtrer pour l'utilisateur
        self.user_history = df_features[df_features['user'] == user_id].copy()
        
        if len(self.user_history) == 0:
            print(f"Aucune donnée trouvée pour l'utilisateur {user_id}")
            return False
        
        # Charger le profil utilisateur
        if not os.path.isabs(profiles_path):
            profiles_path = os.path.join(SCRIPT_DIR, profiles_path)
        
        if os.path.exists(profiles_path):
            df_profiles = pd.read_csv(profiles_path)
            user_profile_data = df_profiles[df_profiles['user_id'] == user_id]
            
            if len(user_profile_data) > 0:
                self.user_profile = user_profile_data.iloc[0].to_dict()
            else:
                print(f"Profil non trouvé pour l'utilisateur {user_id}, valeurs calculées depuis l'historique")
                self.user_profile = None
        else:
            print(f"Fichier de profils non trouvé: {profiles_path}")
            self.user_profile = None
        
        return True
    
    def predict_fatigue_on_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prédit la fatigue sur les données avec le classificateur.
        
        Args:
            data: DataFrame avec les features
            
        Returns:
            Array de prédictions (0 ou 1) et probabilités
        """
        # Sélectionner et convertir en numérique; imputer les NaN par médiane
        features_df = data[self.classifier.FEATURE_NAMES].apply(pd.to_numeric, errors='coerce')
        features_df = features_df.fillna(features_df.median())
        features_clean = features_df.values.astype(float)
        
        # Prédictions
        predictions = self.classifier.predict(features_clean)
        probabilities = self.classifier.predict_proba(features_clean)[:, 1]  # Proba classe 1 (fatigué)
        
        return predictions, probabilities
    
    def analyze_last_session(self) -> Dict:
        """
        Analyse la dernière séance d'entraînement avec prédictions du classificateur.
        
        Returns:
            Dictionnaire avec les statistiques de la dernière séance
        """
        if self.user_history is None or len(self.user_history) == 0:
            return {}
        
        # Si workout_id existe, regrouper par workout_id
        if 'workout_id' in self.user_history.columns:
            # Prendre le dernier workout_id (la séance la plus récente)
            unique_workouts = self.user_history['workout_id'].unique()
            last_workout_id = unique_workouts[-1]  # Dernier workout
            
            # Toutes les données de cette séance (start + sport + recovery)
            data = self.user_history[self.user_history['workout_id'] == last_workout_id].copy()
            
            # Identifier l'activité principale (souvent 'sport' est l'activité clé)
            activity_counts = data['activity'].value_counts()
            main_activity = activity_counts.index[0] if len(activity_counts) > 0 else 'unknown'
            
        else:
            # Rétrocompatibilité: si pas de workout_id, prendre toutes les données du user
            # (considérées comme une seule séance historique)
            data = self.user_history.copy()
            activity_counts = data['activity'].value_counts()
            main_activity = activity_counts.index[0] if len(activity_counts) > 0 else 'unknown'
        
        # Prédire la fatigue avec le classificateur
        predictions, probabilities = self.predict_fatigue_on_data(data)
        
        # Statistiques de la séance (convertir en numérique si nécessaire)
        hr_series = pd.to_numeric(data['ecg_hr_bpm'], errors='coerce')
        hrv_sdnn_series = pd.to_numeric(data['ecg_hrv_sdnn'], errors='coerce')
        hrv_rmssd_series = pd.to_numeric(data['ecg_hrv_rmssd'], errors='coerce')
        emg_rms_series = pd.to_numeric(data['emg_rms'], errors='coerce')
        resp_rate_series = pd.to_numeric(data['resp_rate_bpm'], errors='coerce') if 'resp_rate_bpm' in data.columns else pd.Series(dtype=float)

        stats = {
            'activity': main_activity,  # Activité principale
            'n_windows': len(data),  # Nombre total de fenêtres dans la séance
            'activities_included': data['activity'].unique().tolist() if 'activity' in data.columns else [],
            'hr_mean': hr_series.mean(),
            'hr_max': hr_series.max(),
            'hr_min': hr_series.min(),
            'hrv_sdnn_mean': hrv_sdnn_series.mean(),
            'hrv_rmssd_mean': hrv_rmssd_series.mean(),
            'emg_rms_mean': emg_rms_series.mean(),
            'resp_rate_mean': resp_rate_series.mean() if 'resp_rate_bpm' in data.columns else None,
            'fatigue_ratio': data['label'].mean() if 'label' in data.columns else None,
            # Prédictions du classificateur
            'fatigue_predicted': predictions.mean(),  # Ratio de fenêtres prédites comme fatiguées
            'fatigue_probability': probabilities.mean(),  # Probabilité moyenne de fatigue
            'fatigue_max_probability': probabilities.max(),  # Pic de fatigue
            'fatigue_trend': self._calculate_fatigue_trend(probabilities)  # Tendance
        }
        
        return stats
    
    def _calculate_fatigue_trend(self, probabilities: np.ndarray) -> str:
        """
        Calcule la tendance de fatigue (croissante, décroissante, stable).
        
        Args:
            probabilities: Array de probabilités de fatigue
            
        Returns:
            'increasing', 'decreasing', ou 'stable'
        """
        if len(probabilities) < 3:
            return 'stable'
        
        # Diviser en 2 moitiés
        mid = len(probabilities) // 2
        first_half_mean = probabilities[:mid].mean()
        second_half_mean = probabilities[mid:].mean()
        
        diff = second_half_mean - first_half_mean
        
        if diff > 0.15:  # Augmentation significative
            return 'increasing'
        elif diff < -0.15:  # Diminution significative
            return 'decreasing'
        else:
            return 'stable'
    
    def evaluate_recovery_status(self) -> Dict:
        """
        Évalue l'état de récupération basé sur la dernière séance et le classificateur.
        
        Returns:
            Dictionnaire avec l'évaluation de récupération
        """
        last_stats = self.analyze_last_session()
        
        if not last_stats:
            return {'status': 'unknown', 'score': 0}
        
        # Score de récupération (0-100) - basé principalement sur le classificateur
        recovery_score = 50  # Score neutre de base
        
        # ===== PRÉDICTION DU CLASSIFICATEUR (poids principal) =====
        fatigue_prob = last_stats.get('fatigue_probability', 0)
        fatigue_trend = last_stats.get('fatigue_trend', 'stable')
        
        # Fatigue prédite élevée = score bas
        if fatigue_prob > 0.7:
            recovery_score -= 30
        elif fatigue_prob > 0.5:
            recovery_score -= 20
        elif fatigue_prob > 0.3:
            recovery_score -= 10
        else:
            recovery_score += 15  # Bonne récupération
        
        # Tendance de fatigue
        if fatigue_trend == 'increasing':
            recovery_score -= 15  # Fatigue croissante = mauvais signe
        elif fatigue_trend == 'decreasing':
            recovery_score += 10  # Récupération en cours
        
        # ===== FACTEURS PHYSIOLOGIQUES COMPLÉMENTAIRES =====
        
        # HRV élevée = bonne récupération
        if last_stats['hrv_sdnn_mean'] > 600:
            recovery_score += 15
        elif last_stats['hrv_sdnn_mean'] > 450:
            recovery_score += 5
        elif last_stats['hrv_sdnn_mean'] < 400:
            recovery_score -= 10
        
        # HR basse = bonne récupération
        if last_stats['hr_mean'] < 65:
            recovery_score += 10
        elif last_stats['hr_mean'] < 75:
            recovery_score += 5
        elif last_stats['hr_mean'] > 85:
            recovery_score -= 10

        # EMG élevée = tension musculaire
        if last_stats['emg_rms_mean'] > 15:
            recovery_score -= 10
        
        # Limiter entre 0 et 100
        recovery_score = max(0, min(100, recovery_score))
        
        # Catégoriser avec ajustements basés sur fatigue
        if recovery_score >= 60:
            status = 'good'
            message = "Bonne récupération. Prêt pour un entraînement intense."
        elif recovery_score >= 30:
            status = 'moderate'
            message = "Récupération modérée. Privilégier un entraînement modéré."
        else:
            status = 'poor'
            message = "Récupération insuffisante. Entraînement léger."
        
        return {
            'status': status,
            'score': recovery_score,
            'message': message,
            'last_activity': last_stats['activity'],
            'last_fatigue_ratio': last_stats.get('fatigue_ratio', 0),
            'fatigue_probability': fatigue_prob,
            'fatigue_trend': fatigue_trend
        }
    
    def recommend_training(self) -> Dict:
        """
        Recommande une séance d'entraînement personnalisée basée sur le classificateur de fatigue.
        
        Returns:
            Dictionnaire avec la recommandation complète
        """
        recovery = self.evaluate_recovery_status()
        last_stats = self.analyze_last_session()
        
        # Déterminer HR_repos et HR_max à partir des données historiques
        # (au lieu de charger depuis un profil fichier)
        if self.user_profile:
            hr_repos = self.user_profile.get('HR_repos', 65)
            hr_max = self.user_profile.get('HR_max', 180)
        else:
            # Calculer à partir des données de l'utilisateur
            hr_repos = last_stats.get('hr_min', 65)
            hr_max = last_stats.get('hr_max', 180)
            # Estimer HR_max avec formule de Karvonen si données insuffisantes
            if hr_max <= hr_repos:
                hr_max = hr_repos + 120  # Valeur conservative pour la réserve
        
        hr_reserve = hr_max - hr_repos
        
        # Métriques pour recommandation précise
        fatigue_prob = last_stats.get('fatigue_probability', 0)
        fatigue_trend = last_stats.get('fatigue_trend', 'stable')
        
        # Recommandation basée sur le score de récupération ET la fatigue prédite
        # 3 niveaux de recommandation
        if recovery['score'] >= 60 and fatigue_prob < 0.5:
            # Entraînement modéré à intense
            intensity_name = "Modéré à Intense"
            hr_target_min = hr_repos + 0.6 * hr_reserve  # 60-80% de la réserve
            hr_target_max = hr_repos + 0.8 * hr_reserve
            duration = 30
            description = "Séance d'entraînement modéré à intense. Bonne forme détectée."
            
        elif recovery['score'] >= 30 and fatigue_prob < 0.7:
            # Entraînement léger
            intensity_name = "Léger"
            hr_target_min = hr_repos + 0.45 * hr_reserve  # 45-60% de la réserve
            hr_target_max = hr_repos + 0.6 * hr_reserve
            duration = 25
            description = "Séance d'entraînement léger. Signes de fatigue modérée, intensité réduite."
            
        else:
            intensity_name = "Récupération Active"
            hr_target_min = hr_repos + 0.3 * hr_reserve  # 30-45% de la réserve
            hr_target_max = hr_repos + 0.45 * hr_reserve
            duration = 20
            description = "Récupération active recommandée. Fatigue importante détectée, priorité à la récupération."
        
        
        return {
            'recovery_status': recovery,
            'intensity': intensity_name,
            'hr_target_range': (int(hr_target_min), int(hr_target_max)),
            'duration': duration,
            'description': description,
            'hr_repos': int(hr_repos),
            'hr_max': int(hr_max),
            'fatigue_probability': fatigue_prob,
            'fatigue_trend': fatigue_trend,
            'recommendations': self._generate_specific_recommendations(recovery['score'], last_stats, fatigue_prob, fatigue_trend),
            'adaptive_warnings': self._generate_adaptive_warnings(fatigue_prob, fatigue_trend)
        }
    
    def _generate_adaptive_warnings(self, fatigue_prob: float, fatigue_trend: str) -> List[str]:
        """Génère des avertissements adaptatifs basés sur la fatigue prédite."""
        warnings = []
        
        if fatigue_prob > 0.7:
            warnings.append("ALERTE: Probabilité de fatigue très élevée ({:.0%})".format(fatigue_prob))
            warnings.append("-> Envisagez de reporter l'entraînement ou de faire uniquement de la récupération")
        elif fatigue_prob > 0.5:
            warnings.append("ATTENTION: Fatigue modérée détectée ({:.0%})".format(fatigue_prob))
            warnings.append("-> Réduisez l'intensité et restez attentif aux signaux corporels")
        
        if fatigue_trend == 'increasing':
            warnings.append("TENDANCE: Fatigue en augmentation au cours de la dernière séance")
            warnings.append("-> L'entraînement proposé est adapté à la baisse pour favoriser la récupération")
        elif fatigue_trend == 'decreasing':
            warnings.append("INFO: Bonne récupération observée au cours de la dernière séance")
        
        return warnings
    
    def _generate_specific_recommendations(self, recovery_score: float, last_stats: Dict, 
                                           fatigue_prob: float, fatigue_trend: str) -> List[str]:
        """Génère des recommandations spécifiques basées sur les données et le classificateur."""
        recs = []
        
        # Recommandations basées sur la probabilité de fatigue
        if fatigue_prob < 0.3:
            recs.append("Faible probabilité de fatigue ({:.0%}) - Excellent état!".format(fatigue_prob))
            recs.append("- Profitez de cette forme pour progresser")
            recs.append("- Incluez des exercices de haute intensité si entraînement intense")
        
        elif fatigue_prob < 0.5:
            recs.append("Probabilité de fatigue modérée ({:.0%}) - État normal".format(fatigue_prob))
            recs.append("- Entraînement classique possible")
            recs.append("- Surveillez l'évolution de votre fatigue pendant l'effort")
        
        elif fatigue_prob < 0.7:
            recs.append("Probabilité de fatigue élevée ({:.0%}) - Soyez prudent".format(fatigue_prob))
            recs.append("- Réduisez l'intensité et la durée prévues")
            recs.append("- Arrêtez si les signaux de fatigue s'intensifient")
        
        else:
            recs.append("Probabilité de fatigue très élevée ({:.0%}) - Repos recommandé".format(fatigue_prob))
            recs.append("- Privilégiez le repos ou récupération très légère")
            recs.append("- Ne forcez pas - écoutez votre corps")
        
        # Recommandations basées sur la tendance
        if fatigue_trend == 'increasing':
            recs.append("Tendance: Fatigue croissante détectée")
            recs.append("- Le corps accumule de la fatigue - attention au surentraînement")
            recs.append("- Intégrez plus de récupération dans votre planning")
        elif fatigue_trend == 'decreasing':
            recs.append("Tendance: Récupération en cours")
            recs.append("- Continuez sur cette lancée avec une charge progressive")
        
        # Recommandations basées sur HRV
        if last_stats and last_stats.get('hrv_sdnn_mean', 0) < 400:
            recs.append("- HRV basse détectée - évitez le surentraînement")
        elif last_stats and last_stats.get('hrv_sdnn_mean', 0) > 700:
            recs.append("HRV excellente - système nerveux bien récupéré")
        
        # Recommandations basées sur EMG
        if last_stats and last_stats.get('emg_rms_mean', 0) > 15:
            recs.append("- Tension musculaire élevée - incluez des étirements/massage")
        
        # Recommandations basées sur l'activité précédente
        if last_stats and last_stats.get('activity') == 'sport' and fatigue_prob > 0.5:
            recs.append("- Séance intense récente + fatigue = besoin de récupération supplémentaire")
        
        return recs
    
    def generate_report(self, user_id: int) -> str:
        """
        Génère un rapport complet pour l'utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Rapport formaté en texte
        """
        if not self.load_user_data(user_id):
            return f"Impossible de charger les données pour l'utilisateur {user_id}"
        
        # Analyse de la dernière séance
        last_session = self.analyze_last_session()
        
        # Recommandation
        recommendation = self.recommend_training()
        
        # Formatage du rapport
        report = []
        report.append("=" * 70)
        report.append(f"RECOMMANDATION D'ENTRAÎNEMENT - USER {user_id}")
        report.append("=" * 70)
        
        # Profil utilisateur
        if self.user_profile:
            report.append("\nPROFIL UTILISATEUR")
            report.append(f"  (Profil chargé depuis données utilisateur)")
        else:
            report.append("\nPROFIL UTILISATEUR")
            report.append(f"  (Pas de profil permanent, basé sur historique)")
        
        # Dernière séance
        report.append("\nDERNIERE SEANCE")
        report.append(f"  Activité: {last_session.get('activity', 'N/A')}")
        report.append(f"  HR moyenne: {last_session.get('hr_mean', 0):.1f} bpm")
        report.append(f"  HR max: {last_session.get('hr_max', 0):.1f} bpm")
        report.append(f"  HRV (SDNN): {last_session.get('hrv_sdnn_mean', 0):.1f}")
        report.append(f"  Fatigue (labels): {last_session.get('fatigue_ratio', 0):.1%}")
        report.append(f"  Fatigue (prédite): {last_session.get('fatigue_predicted', 0):.1%}")
        report.append(f"  Probabilité fatigue: {last_session.get('fatigue_probability', 0):.1%}")
        report.append(f"  Tendance fatigue: {last_session.get('fatigue_trend', 'N/A')}")
        
        # État de récupération
        recovery = recommendation['recovery_status']
        report.append("\nETAT DE RECUPERATION (Analyse par IA)")
        report.append(f"  Score: {recovery['score']}/100")
        report.append(f"  Statut: {recovery['status'].upper()}")
        report.append(f"  {recovery['message']}")
        report.append(f"  Probabilité fatigue: {recovery.get('fatigue_probability', 0):.1%}")
        report.append(f"  Tendance: {recovery.get('fatigue_trend', 'N/A')}")
        
        # Recommandation
        report.append("\nPROCHAINE SEANCE RECOMMANDEE (Adaptative)")
        report.append(f"  Intensité: {recommendation['intensity']}")
        report.append(f"  Durée: {recommendation['duration']} minutes")
        report.append(f"  Zone HR cible: {recommendation['hr_target_range'][0]}-{recommendation['hr_target_range'][1]} bpm")
        report.append(f"  Description: {recommendation['description']}")
        
        # Avertissements adaptatifs
        if recommendation.get('adaptive_warnings'):
            report.append("\nAVERTISSEMENTS ADAPTATIFS")
            for warning in recommendation['adaptive_warnings']:
                report.append(f"  {warning}")
        
        # Recommandations spécifiques
        report.append("\nRECOMMANDATIONS SPECIFIQUES")
        for rec in recommendation['recommendations']:
            report.append(f"  {rec}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def main():
    """Fonction principale pour tester le recommandeur."""
    import sys
    
    # ID de l'utilisateur
    user_id = 0  # User0
    
    print(f"\nAnalyse de l'utilisateur {user_id}...\n")
    
    try:
        # Créer le recommandeur
        recommender = TrainingRecommender()
        
        # Générer le rapport
        report = recommender.generate_report(user_id)
        
        print(report)
        
    except FileNotFoundError as e:
        print(f"Erreur: Fichier non trouvé - {e}")
        print("Assurez-vous que fatigue_model.pkl et features_dataset.csv existent.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
