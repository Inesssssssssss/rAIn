# rAIn 💧 - Entraînement Adaptatif par IA

Application d'entraînement de course adapté par IA.

**GitHub**: https://github.com/votre-username/projet_rain

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

### 1. Lancer l'application principale
```bash
python main.py
```
Lance l'interface graphique pour l'entraînement adaptatif.

---

## Développement (optionnel)

> **Note** : Les données et modèles sont déjà générés et inclus dans le projet. Il n'est pas nécessaire de lancer les commandes suivantes sauf si vous souhaitez régénérer les données ou modifier les modèles.

### 2. Traiter les données brutes
```bash
python ai/process_data.py
```
Génère le fichier `features_dataset.csv` à partir des données dans le dossier `data/`.

### 3. Augmenter les données avec des utilisateurs synthétiques
```bash
python ai/user_simulation.py
```
Crée le fichier `features_dataset_augmented.csv` avec des données simulées supplémentaires.

### 4. Entraîner et évaluer le classificateur de fatigue
```bash
python ai/fatigue_classifier.py
```
Entraîne le modèle Random Forest et affiche les métriques de performance.

### 5. Tester le système de recommandation
```bash
python ai/training_recommender.py
```
Simule des recommandations d'entraînement adaptatif.
