from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QLabel, QPushButton,
    QVBoxLayout, QFormLayout,
    QSpinBox, QComboBox, QTextEdit
)

from ai.training_engine import generate_training_plan


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("rAIn")
        self.setMinimumSize(500, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()

        # ===== Formulaire utilisateur =====
        form_layout = QFormLayout()

        self.age_input = QSpinBox()
        self.age_input.setRange(10, 100)
        self.age_input.setValue(25)

        self.level_input = QComboBox()
        self.level_input.addItems(["Débutant", "Intermédiaire", "Avancé"])

        self.goal_input = QComboBox()
        self.goal_input.addItems([
            "Remise en forme",
            "Amélioration endurance",
            "Perte de poids"
        ])

        form_layout.addRow("Âge :", self.age_input)
        form_layout.addRow("Niveau :", self.level_input)
        form_layout.addRow("Objectif :", self.goal_input)

        # ===== Bouton génération =====
        self.generate_button = QPushButton("Générer mon entraînement")
        self.generate_button.clicked.connect(self.on_generate_clicked)

        # ===== Zone résultat =====
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)

        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.generate_button)
        main_layout.addWidget(QLabel("Plan d'entraînement proposé :"))
        main_layout.addWidget(self.result_area)

        central_widget.setLayout(main_layout)

    def on_generate_clicked(self):
        age = self.age_input.value()
        level = self.level_input.currentText()
        goal = self.goal_input.currentText()

        plan = generate_training_plan(
            age=age,
            level=level,
            goal=goal
        )

        self.result_area.setText(plan)
