# AI-ADAPT – Prétraitement ML

Ce projet propose un pipeline simple pour transformer des enregistrements OpenSignals (.txt) en features exploitables par des modèles de Machine Learning (fenêtrage + features basiques + normalisation optionnelle).

## Prérequis

- Python 3.9+
- Packages: voir `requirements.txt`.

Installez les dépendances:

```bash
pip install -r requirements.txt
```

## Exporter des features depuis un dossier OpenSignals

Utilisez le runner CLI:

```bash
python other/preprocess_runner.py --input "C:/Users/inesr/Documents/OpenSignals (r)evolution/files/ai_adapt" --output "other/features.csv" --window 5
```

Options utiles:
- `--window 5` : taille de fenêtre en secondes.
- `--no-normalize` : désactive la normalisation (z-score).

Les features calculées par fenêtre pour chaque canal comprennent: mean, std, min, max, median, rms.

## API Python

Vous pouvez aussi appeler directement depuis Python:

```python
from ai.training_engine import preprocess_folder_to_csv

preprocess_folder_to_csv(
    folder_path=r"C:/Users/inesr/Documents/OpenSignals (r)evolution/files/ai_adapt",
    output_csv=r"other/features.csv",
    window_seconds=5,
    normalize=True,
)
```

Ensuite, chargez `other/features.csv` dans vos notebooks ou scripts ML.
