# rAIn - Entraînement Adaptatif par IA

Application d'entraînement de course intelligent qui s'adapte à votre fatigue physiologique en temps réel grâce à la lecture de signaux biophysiologiques (ECG, EMG, respiration).

## 🚀 Démarrage rapide

### Prérequis
- Python 3.8+
- pip ou conda

### Installation

1. **Cloner/télécharger le projet**
```bash
cd projet_rain
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Lancer l'application**
```bash
python main.py
```

L'interface graphique se lancera automatiquement.

---

## 📋 Structure du projet

```
projet_rain/
├── main.py                 # Point d'entrée principal
├── requirements.txt        # Dépendances Python
│
├── ai/                     # Modèles et traitement IA
│   ├── fatigue_classifier.py      # Classificateur Random Forest
│   ├── training_recommender.py    # Système de recommandation
│   └── process_data.py            # Traitement des données brutes
│
├── ui/                     # Interface utilisateur
│   ├── main_window.py             # Interface graphique PySide6
│   └── live_stream.py             # Lecture des capteurs LSL
│
├── data/                   # Données d'entraînement statiques
│
├── models/                 # Modèles ML entraînés
│   ├── fatigue_model.pkl
│   └── fatigue_scaler.pkl
│
└── output/                 # Fichiers générés à l'exécution
    ├── user_profiles.csv
    ├── features/          # Datasets de features
    └── sessions/          # Logs d'entraînement
```

---

## 💻 Utilisation

### Mode utilisateur existant
1. Lancer `python main.py`
2. Entrer votre ID utilisateur
3. L'application analyse vos données passées
4. Cliquer sur "Commencer la séance"
5. Connecter vos capteurs (LSL) et cliquer "Démarrer Live"
6. Suivre les phases d'entraînement indiquées

### Créer un nouveau compte
1. Lancer `python main.py`
2. Cliquer "Nouvel Utilisateur"
3. Entrer votre profil (âge, sexe, niveau)
4. Une séance d'initialisation (10 min) se lance pour collecter vos données
5. Les données sont enregistrées automatiquement

---

## ⚙️ Configuration recommandée

### Capteurs supportés
- **Bitalino** ou autre appareil compatible LSL (Lab Streaming Layer)
- Canaux attendus:
  - Canal 1: ECG (fréquence cardiaque)
  - Canal 2: EMG (jambe)
  - Canal 3: Respiration

### Système LSL
L'application utilise **Lab Streaming Layer (LSL)** pour lire les capteurs en temps réel:
- Assurez-vous que pylsl est installé (`pip install pylsl`)
- Votre appareil de capteurs doit diffuser sur LSL avec le nom 'OpenSignals'

---

## 📊 Fichiers importants

- **output/user_profiles.csv**: Profils des utilisateurs (âge, sexe, niveau, paramètres physiologiques)
- **output/features/**: Données d'entraînement (features extraites des signaux)
- **output/sessions/**: Logs de chaque séance d'entraînement
- **models/**: Modèles ML sauvegardés pour la prédiction de fatigue

---

## 🔧 Dépendances principales

- **PySide6**: Interface graphique
- **numpy, pandas**: Traitement de données
- **scikit-learn, joblib**: Machine Learning
- **neurokit2**: Analyse de signaux biophysiologiques
- **pylsl**: Lecture de capteurs en temps réel
- **scipy**: Traitement du signal

---

## 📝 Notes

- Les données utilisateur sont stockées dans `output/user_profiles.csv`
- Les features extraites des capteurs sont dans `output/features/`
- Les modèles ML sont entraînés automatiquement et sauvegardés dans `models/`
- Chaque séance d'entraînement génère des logs dans `output/sessions/`

---

## ❓ Dépannage

**"Impossible de trouver les modèles"**
- Vérifiez que `models/fatigue_model.pkl` et `fatigue_scaler.pkl` existent
- Sinon, réentraînez le modèle avec les données disponibles

**"Impossible de se connecter aux capteurs"**
- Vérifiez que LSL est installé: `pip install pylsl`
- Assurez-vous que vos capteurs diffusent sur LSL avec le nom 'OpenSignals'

**"Aucune donnée trouvée pour l'utilisateur"**
- Créez d'abord un compte utilisateur avec une séance d'initialisation
- Ou vérifiez que `output/features/features_dataset.csv` contient des données
