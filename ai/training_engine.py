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

