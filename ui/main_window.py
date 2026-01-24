import sys
import os
import json
import csv
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QPushButton, QSpinBox, QTextEdit, 
                                QProgressBar, QGroupBox, QMessageBox, QStackedWidget,
                                QLineEdit, QComboBox)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
import threading
import time
from PySide6.QtCore import QTimer
import pandas as pd
import numpy as np

# Ajouter le chemin du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai.training_recommender import TrainingRecommender
from ui.live_stream import LiveLSLReader
from ai.process_data import process_file




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("rAIn - Entraînement Adaptatif IA")
        self.setMinimumSize(900, 800)
        
        self.current_recommendation = None
        self.elapsed_seconds = 0
        self.phase_elapsed = 0
        self.phase_type = "effort"
        self.work_duration_sec = 120
        self.rest_duration_sec = 60
        self.is_running = False
        self.is_paused = False
        self.timer_thread = None
        self.target_duration = 0
        self.current_user_id = None
        self.is_new_user = False
        self.initial_phases = None
        self.current_phase_index = 0
        
        # Chemin des fichiers de profil utilisateur
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.project_root, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'features'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'sessions'), exist_ok=True)
        self.user_profiles_path = os.path.join(self.output_dir, 'user_profiles.csv')
        self.features_path = os.path.join(self.output_dir, 'features', 'features_dataset.csv')
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        title_label = QLabel("💧 rAIn - entraînement de course par IA")
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)

        self.stacked = QStackedWidget()
        main_layout.addWidget(self.stacked)

        self.login_page = self.build_login_page()
        self.intro_page = self.build_intro_page()
        self.session_page = self.build_session_page()
        self.stacked.addWidget(self.login_page)
        self.stacked.addWidget(self.intro_page)
        self.stacked.addWidget(self.session_page)
        self.stacked.setCurrentWidget(self.login_page)

        self.live_reader = None
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(200)
        self.live_timer.timeout.connect(self.update_live_values)

        # Appliquer le thème blanc/bleu
        self.apply_theme()

    def apply_theme(self):
        """Applique un thème global blanc/bleu à l'interface."""
        theme_css = """
        QWidget {
            background-color: #ffffff;
            color: #1f2937;
        }
        QGroupBox {
            background-color: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 6px;
            margin-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: #1E3A8A;
            font-weight: bold;
        }
        QLabel {
            color: #1f2937;
        }
        QPushButton {
            background-color: #1E88E5;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565C0;
        }
        QPushButton:disabled {
            background-color: #93c5fd;
            color: #e5e7eb;
        }
        QLineEdit, QSpinBox, QComboBox, QTextEdit {
            background-color: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 6px;
        }
        QProgressBar {
            border: 1px solid #dbeafe;
            border-radius: 6px;
            text-align: center;
            height: 18px;
        }
        QProgressBar::chunk {
            background-color: #1E88E5;
            border-radius: 6px;
        }
        """
        self.setStyleSheet(theme_css)

    def build_login_page(self):
        """Page de connexion/inscription"""
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        # Espacement
        layout.addStretch()

        # Titre
        title = QLabel("Bienvenue dans rAIN")
        title_font = QFont("Arial", 16, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Gestionnaire d'entraînement adaptatif par IA")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 11pt;")
        layout.addWidget(subtitle)

        layout.addSpacing(30)

        # Groupe utilisateur existant
        existing_group = QGroupBox("Utilisateur Existant")
        existing_layout = QVBoxLayout()
        existing_group.setLayout(existing_layout)

        existing_layout.addWidget(QLabel("Entrez votre ID utilisateur:"))
        self.existing_user_id = QSpinBox()
        self.existing_user_id.setRange(0, 100)
        self.existing_user_id.setValue(0)
        self.existing_user_id.setMinimumHeight(40)
        existing_layout.addWidget(self.existing_user_id)

        login_btn = QPushButton("Se connecter")
        login_btn.setMinimumHeight(40)
        login_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; font-size: 12pt; border-radius: 5px; }")
        login_btn.clicked.connect(self.on_existing_user_login)
        existing_layout.addWidget(login_btn)

        layout.addWidget(existing_group)

        # Séparation
        separator = QLabel("OU")
        separator.setAlignment(Qt.AlignCenter)
        separator.setStyleSheet("color: #95a5a6; font-weight: bold; margin: 20px 0px;")
        layout.addWidget(separator)

        # Groupe nouvel utilisateur
        new_group = QGroupBox("Nouvel Utilisateur")
        new_layout = QVBoxLayout()
        new_group.setLayout(new_layout)

        new_layout.addWidget(QLabel("Profil utilisateur:"))

        profile_h_layout = QHBoxLayout()
        profile_h_layout.addWidget(QLabel("Âge:"))
        self.new_user_age = QSpinBox()
        self.new_user_age.setRange(15, 80)
        self.new_user_age.setValue(25)
        profile_h_layout.addWidget(self.new_user_age)

        profile_h_layout.addSpacing(20)
        profile_h_layout.addWidget(QLabel("Sexe:"))
        self.new_user_sex = QComboBox()
        self.new_user_sex.addItems(["M", "F"])
        profile_h_layout.addWidget(self.new_user_sex)

        profile_h_layout.addSpacing(20)
        profile_h_layout.addWidget(QLabel("Niveau:"))
        self.new_user_fitness = QComboBox()
        self.new_user_fitness.addItems(["débutant", "intermédiaire", "avancé"])
        profile_h_layout.addWidget(self.new_user_fitness)
        profile_h_layout.addStretch()

        new_layout.addLayout(profile_h_layout)

        signup_btn = QPushButton("Créer un compte et commencer")
        signup_btn.setMinimumHeight(40)
        signup_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; font-size: 12pt; border-radius: 5px; }")
        signup_btn.clicked.connect(self.on_new_user_signup)
        new_layout.addWidget(signup_btn)

        layout.addWidget(new_group)

        layout.addStretch()
        return page

    def on_existing_user_login(self):
        """Connexion utilisateur existant"""
        user_id = self.existing_user_id.value()
        
        # Vérifier que l'utilisateur existe
        if os.path.exists(self.user_profiles_path):
            df = pd.read_csv(self.user_profiles_path)
            if user_id not in df['user_id'].values:
                QMessageBox.warning(self, "Erreur", f"L'utilisateur {user_id} n'existe pas.")
                return
        
        self.current_user_id = user_id
        self.is_new_user = False
        self.stacked.setCurrentWidget(self.intro_page)
        # Récupérer la recommandation automatiquement
        self.auto_analyze_user()

    def on_new_user_signup(self):
        """Inscription nouvel utilisateur"""
        # Générer un nouvel ID
        new_user_id = self.get_next_user_id()
        
        age = self.new_user_age.value()
        sex = self.new_user_sex.currentText()
        fitness = self.new_user_fitness.currentText()
        
        # Générer des paramètres physiologiques aléatoires réalistes
        hr_repos = np.random.uniform(50, 90)
        hr_max = np.random.uniform(170, 200)
        rr_repos = np.random.uniform(10, 30)
        
        # Ajouter au fichier CSV user_profiles
        new_profile = {
            'user_id': new_user_id,
            'age': age,
            'sex': sex,
            'fitness': fitness,
            'HR_repos': round(hr_repos, 1),
            'HR_max': round(hr_max, 1),
            'RR_repos': round(rr_repos, 1)
        }
        
        # Ajouter à user_profiles.csv
        if os.path.exists(self.user_profiles_path):
            df_profiles = pd.read_csv(self.user_profiles_path)
            df_profiles = pd.concat([df_profiles, pd.DataFrame([new_profile])], ignore_index=True)
        else:
            df_profiles = pd.DataFrame([new_profile])
        
        df_profiles.to_csv(self.user_profiles_path, index=False)
        
        # Créer l'entraînement initial (10 min: 3 min marche, 5 min course, 2 min marche)
        self.create_initial_training_session(new_user_id, age, sex, fitness)
        
        self.current_user_id = new_user_id
        self.is_new_user = True
        
        QMessageBox.information(self, "Compte créé", 
            f"Compte créé avec succès!\nVotre ID utilisateur: {new_user_id}\n\n"
            f"Vous allez commencer une séance d'entraînement initiale de 10 minutes "
            f"pour collecter vos données de base.")
        
        self.stacked.setCurrentWidget(self.intro_page)
        self.show_initial_training_for_new_user()

    def get_next_user_id(self) -> int:
        """Obtient le prochain ID utilisateur disponible"""
        if os.path.exists(self.user_profiles_path):
            df = pd.read_csv(self.user_profiles_path)
            return int(df['user_id'].max()) + 1
        return 0

    def create_initial_training_session(self, user_id: int, age: int, sex: str, fitness: str):
        """Crée une première session d'entraînement pour collecter les données de base"""
        initial_session = {
            'user': user_id,
            'age': age,
            'sex': sex,
            'fitness': fitness,
            'duration': 10,  # 10 minutes
            'phases': [
                {'type': 'walk', 'duration': 3},      # 3 min marche
                {'type': 'run', 'duration': 5},       # 5 min course
                {'type': 'walk', 'duration': 2}       # 2 min marche
            ],
            'is_initial': True
        }
        
        # Sauvegarder dans un fichier JSON ou dans la structure features
        # Pour maintenant, on stocke simplement en mémoire
        self.initial_training = initial_session

    def show_initial_training_for_new_user(self):
        """Affiche le plan d'entraînement initial pour un nouvel utilisateur"""
        self.result_area.clear()
        
        text = "=" * 70 + "\n"
        text += "PLAN D'ENTRAÎNEMENT INITIAL\n"
        text += "=" * 70 + "\n\n"
        text += "Bienvenue! Vous allez débuter par une séance d'entraînement de 10 minutes\n"
        text += "destinée à collecter vos premières données physiologiques.\n\n"
        text += "PROGRAMME:\n"
        text += "├─ 3 minutes de marche (échauffement)\n"
        text += "├─ 5 minutes de course (phase principale)\n"
        text += "└─ 2 minutes de marche (récupération)\n\n"
        text += "Cette séance servira de base pour adapter vos futurs entraînements.\n\n"
        text += "INSTRUCTIONS:\n"
        text += "• Assurez-vous que le système LSL est connecté (Live Stream)\n"
        text += "• Suivez les indications de phase affichées à l'écran\n"
        text += "• Vous pouvez ajuster l'intensité selon vos sensations\n"
        text += "• Les données seront enregistrées automatiquement\n"
        
        self.result_area.setPlainText(text)
        
        # Préparer la séance
        self.current_recommendation = {
            'intensity': 'low',
            'duration': 10,
            'hr_target_range': [100, 140],
            'description': 'Entraînement initial pour collecte de données',
            'recovery_status': {
                'status': 'initial',
                'score': 50,
                'message': 'Première séance - pas d\'historique',
                'fatigue_probability': 0.0,
                'fatigue_trend': 'N/A'
            },
            'recommendations': [],
            'adaptive_warnings': [],
            'is_initial': True
        }
        
        self.setup_timer()
        self.begin_btn.setEnabled(True)
        self.begin_btn.show()

    def auto_analyze_user(self):
        """Analyse automatiquement l'utilisateur existant"""
        try:
            self.result_area.clear()
            self.result_area.setPlainText("Analyse en cours...\n")
            self.begin_btn.setEnabled(False)
            self.begin_btn.hide()
            
            recommender = TrainingRecommender()
            
            if not recommender.load_user_data(self.current_user_id):
                QMessageBox.warning(self, "Erreur", f"Aucune donnée pour l'utilisateur {self.current_user_id}")
                self.result_area.clear()
                self.begin_btn.setEnabled(False)
                self.begin_btn.hide()
                return
            
            self.current_recommendation = recommender.recommend_training()
            
            self.display_recommendation()
            self.setup_timer()
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur: {e}")
            self.result_area.clear()

    def build_intro_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        rec_group = QGroupBox("Recommandation")
        rec_layout = QVBoxLayout()
        rec_group.setLayout(rec_layout)

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setMinimumHeight(250)
        self.result_area.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        rec_layout.addWidget(self.result_area)

        layout.addWidget(rec_group)

        buttons_layout = QHBoxLayout()
        
        back_btn = QPushButton("Retour")
        back_btn.setStyleSheet("QPushButton { background-color: #95a5a6; color: white; font-weight: bold; padding: 10px 20px; }")
        back_btn.clicked.connect(self.go_back_to_login)
        buttons_layout.addWidget(back_btn)

        self.begin_btn = QPushButton("Commencer la séance")
        self.begin_btn.setEnabled(False)
        self.begin_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 10px 20px; }")
        self.begin_btn.clicked.connect(self.show_session_page)
        self.begin_btn.hide()
        buttons_layout.addWidget(self.begin_btn)
        
        buttons_layout.addStretch()

        layout.addLayout(buttons_layout)

        layout.addStretch()
        return page

    def go_back_to_login(self):
        """Retour à l'écran de connexion"""
        self.stacked.setCurrentWidget(self.login_page)

    def build_session_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        live_group = QGroupBox("Données Physiologiques (Live)")
        live_layout = QVBoxLayout()
        live_group.setLayout(live_layout)

        self.live_status_label = QLabel("Statut: Arrêté")
        self.live_status_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(self.live_status_label)

        self.live_labels = [QLabel("Ch1: --"), QLabel("Ch2: --"), QLabel("Ch3: --")] 
        for lbl in self.live_labels:
            lbl.setAlignment(Qt.AlignCenter)
            live_layout.addWidget(lbl)

        self.live_hr_label = QLabel("HR: -- bpm")
        self.live_hrv_label = QLabel("HRV (RMSSD): -- ms")
        self.live_resp_label = QLabel("Resp: -- bpm")
        self.live_emg_label = QLabel("EMG RMS: --")
        for lbl in (self.live_hr_label, self.live_hrv_label, self.live_resp_label, self.live_emg_label):
            lbl.setAlignment(Qt.AlignCenter)
            live_layout.addWidget(lbl)

        live_btns = QHBoxLayout()
        self.live_start_btn = QPushButton("Démarrer Live")
        self.live_start_btn.clicked.connect(self.start_live)
        self.live_stop_btn = QPushButton("Arrêter Live")
        self.live_stop_btn.clicked.connect(self.stop_live)
        self.live_stop_btn.setEnabled(False)
        live_btns.addWidget(self.live_start_btn)
        live_btns.addWidget(self.live_stop_btn)
        live_layout.addLayout(live_btns)


        layout.addWidget(live_group)

        timer_group = QGroupBox("Session d'Entraînement")
        timer_layout = QVBoxLayout()
        timer_group.setLayout(timer_layout)

        self.timer_label = QLabel("00:00")
        timer_font = QFont("Arial", 48, QFont.Bold)
        self.timer_label.setFont(timer_font)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("color: #3498db;")
        timer_layout.addWidget(self.timer_label)

        self.duration_label = QLabel("Durée recommandée: -- minutes")
        self.duration_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.duration_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(700)
        timer_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("0%")
        self.progress_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.progress_label)

        self.zones_label = QLabel("Zone HR: -- bpm")
        zones_font = QFont("Arial", 11, QFont.Bold)
        self.zones_label.setFont(zones_font)
        self.zones_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.zones_label)

        self.phase_label = QLabel("Phase: --")
        self.phase_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.phase_label)

        self.phase_remaining_label = QLabel("Temps restant phase: --")
        self.phase_remaining_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.phase_remaining_label)

        controls_layout = QHBoxLayout()

        self.start_pause_btn = QPushButton("Démarrer")
        self.start_pause_btn.clicked.connect(self.toggle_start_pause)
        self.start_pause_btn.setEnabled(False)
        self.start_pause_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 10px 20px; min-width: 120px; }")
        controls_layout.addWidget(self.start_pause_btn)

        self.stop_btn = QPushButton("Arrêter")
        self.stop_btn.clicked.connect(self.stop_session)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #95a5a6; color: white; font-weight: bold; padding: 10px 20px; min-width: 120px; }")
        controls_layout.addWidget(self.stop_btn)

        self.finish_btn = QPushButton("Terminer")
        self.finish_btn.clicked.connect(self.finish_session)
        self.finish_btn.setEnabled(False)
        self.finish_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; padding: 10px 20px; min-width: 120px; }")
        controls_layout.addWidget(self.finish_btn)

        timer_layout.addLayout(controls_layout)

        layout.addWidget(timer_group)
        layout.addStretch()
        return page

    def show_session_page(self):
        if self.current_recommendation is None:
            QMessageBox.warning(self, "Séance", "Analysez d'abord l'utilisateur pour obtenir une recommandation.")
            return
        self.stacked.setCurrentWidget(self.session_page)
        
    def display_recommendation(self):
        rec = self.current_recommendation
        recovery = rec['recovery_status']
        
        text = "=" * 70 + "\n"
        text += "ANALYSE ET RECOMMANDATION\n"
        text += "=" * 70 + "\n\n"
        text += f"État: {recovery['status'].upper()} ({recovery['score']}/100)\n"
        text += f"Fatigue: {recovery.get('fatigue_probability', 0):.1%} | Tendance: {recovery.get('fatigue_trend', 'N/A')}\n"
        text += f"{recovery['message']}\n\n"
        text += f"Durée: {rec['duration']} min\n"
        text += f"Zone HR: {rec['hr_target_range'][0]}-{rec['hr_target_range'][1]} bpm\n\n"
        text += f"{rec['description']}\n\n"
        
        if rec.get('adaptive_warnings'):
            text += "AVERTISSEMENTS:\n"
            for w in rec['adaptive_warnings']:
                text += f"• {w}\n"
            text += "\n"
        
        text += "RECOMMANDATIONS:\n"
        for r in rec['recommendations']:
            text += f"• {r}\n"
        
        self.result_area.clear()
        self.result_area.setPlainText(text)
        if hasattr(self, 'begin_btn'):
            self.begin_btn.setEnabled(True)
            self.begin_btn.show()
    
    def setup_timer(self):
        rec = self.current_recommendation
        recommended_seconds = rec['duration'] * 60
        self.target_duration = min(recommended_seconds, 30 * 60)
        
        self.duration_label.setText(f"Durée recommandée: {min(rec['duration'], 30)} minutes")
        
        hr_min, hr_max = rec['hr_target_range']
        self.zones_label.setText(f"Zone HR cible: {hr_min}-{hr_max} bpm")
        
        self.progress_bar.setMaximum(self.target_duration)

        # Gérer le cas de l'entraînement initial (phases fixes)
        if rec.get('is_initial'):
            # Entraînement initial: 3 min marche, 5 min course, 2 min marche
            """
            self.initial_phases = [
                {'type': 'walk', 'duration': 3 * 60, 'label': 'Marche (échauffement)'},
                {'type': 'run', 'duration': 5 * 60, 'label': 'Course (phase principale)'},
                {'type': 'walk', 'duration': 2 * 60, 'label': 'Marche (récupération)'}
            ]
            """
            self.initial_phases = [
                {'type': 'walk', 'duration': 10, 'label': 'Marche (échauffement)'},
                {'type': 'run', 'duration': 10, 'label': 'Course (phase principale)'},
                {'type': 'walk', 'duration': 10, 'label': 'Marche (récupération)'}
            ]
            self.current_phase_index = 0
            self.phase_label.setText("Phase: Marche (échauffement)")
            self.phase_remaining_label.setText(f"Temps restant phase: {3:02d}:00")
            self.phase_label.show()
            self.phase_remaining_label.show()
        else:
            # Entraînement normal: pas de phases, juste la durée totale
            self.initial_phases = None
            self.phase_label.hide()
            self.phase_remaining_label.hide()
        
        self.elapsed_seconds = 0
        self.phase_elapsed = 0
        
        self.start_pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.finish_btn.setEnabled(False)
    
    def toggle_start_pause(self):
        if not self.is_running:
            self.start_session()
        else:
            if self.is_paused:
                self.resume_session()
            else:
                self.pause_session()
    
    def start_session(self):
        self.start_live(record=True)

        self.is_running = True
        self.is_paused = False
        self.elapsed_seconds = 0
        self.phase_elapsed = 0
        
        self.start_pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.finish_btn.setEnabled(True)
        
        self.timer_thread = threading.Thread(target=self.run_timer, daemon=True)
        self.timer_thread.start()
    
    def pause_session(self):
        self.is_paused = True
        self.start_pause_btn.setText("Reprendre")
    
    def resume_session(self):
        self.is_paused = False
        self.start_pause_btn.setText("Pause")
    
    def stop_session(self):
        self.is_running = False
        self.is_paused = False
        self.elapsed_seconds = 0
        self.phase_elapsed = 0
        
        self.timer_label.setText("00:00")
        self.timer_label.setStyleSheet("color: #3498db;")
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")
        
        # Masquer les labels de phase
        self.phase_label.hide()
        self.phase_remaining_label.hide()
        
        self.start_pause_btn.setText("Démarrer")
        self.start_pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.finish_btn.setEnabled(False)
    
    def finish_session(self):
        self.is_running = False
        
        minutes = self.elapsed_seconds // 60
        seconds = self.elapsed_seconds % 60
        saved_path = self.stop_live(save=True)
        extra = f"\nDonnées live enregistrées: {saved_path}" if saved_path else "\nAucune donnée live enregistrée."
        
        QMessageBox.information(self, "Séance terminée", f"Félicitations!\n\nDurée: {minutes:02d}:{seconds:02d}{extra}")
        self.stop_session()

    def start_live(self, record=False):
        try:
            if self.live_reader is None:
                self.live_reader = LiveLSLReader(stream_name='OpenSignals', forced_mapping={'ecg': 1, 'emg': 2, 'resp': 3}, timeout=10.0)
            already_running = self.live_reader._thread is not None and self.live_reader._thread.is_alive()
            if not already_running:
                self.live_reader.start()
            if record:
                self.live_reader.start_recording()
            self.live_status_label.setText("Statut: Connecté")
            self.live_start_btn.setEnabled(False)
            self.live_stop_btn.setEnabled(True)
            if not self.live_timer.isActive():
                self.live_timer.start()
        except RuntimeError as e:
            error_msg = str(e)
            if "No LSL stream found" in error_msg:
                QMessageBox.warning(self, "Live", "Le flux OpenSignals n'a pas pu être récupéré.\nVérifiez que l'appareil est connecté et que la transmission est activée.")
            else:
                QMessageBox.critical(self, "Live", f"Impossible de démarrer le live: {e}")
            self.live_status_label.setText("Statut: Erreur de connexion")
            self.live_start_btn.setEnabled(True)
            self.live_stop_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Live", f"Impossible de démarrer le live: {e}")
            self.live_status_label.setText("Statut: Erreur")
            self.live_start_btn.setEnabled(True)
            self.live_stop_btn.setEnabled(False)

    def stop_live(self, save=False):
        try:
            saved_path = None
            if self.live_reader:
                if save:
                    saved_path = self._save_live_recording()
                else:
                    self.live_reader.stop_recording()
                self.live_reader.stop()
            self.live_status_label.setText("Statut: Arrêté")
            self.live_start_btn.setEnabled(True)
            self.live_stop_btn.setEnabled(False)
            self.live_timer.stop()
            for lbl in self.live_labels:
                lbl.setText(lbl.text().split(':')[0] + ": --")
            return saved_path
        except Exception as e:
            QMessageBox.critical(self, "Live", f"Erreur lors de l'arrêt du live: {e}")
            return None

    def _save_live_recording(self):
        if not self.live_reader:
            return None
        samples = self.live_reader.stop_recording()
        if not samples:
            return None
        user_id = self.current_user_id
        sessions_dir = os.path.join(self.output_dir, 'sessions')
        os.makedirs(sessions_dir, exist_ok=True)
        workout_id = self._next_workout_id(sessions_dir, user_id)
        fname = f"sport_user{user_id}_workout{workout_id}.txt"
        path = os.path.join(sessions_dir, fname)

        srate = int(self.live_reader.latest.get('srate') or 50)
        header_obj = {
            "LiveLSL": {
                "position": 0,
                "device": "lsl_stream",
                "device name": "LiveLSL",
                "device connection": "lsl",
                "sampling rate": srate,
                "resolution": [4, 1, 1, 1, 1, 10, 10, 10, 10, 6, 6],
                "firmware version": 0,
                "comments": "recorded via rAIn",
                "keywords": "",
                "mode": 0,
                "sync interval": 2,
                "date": time.strftime("%Y-%m-%d"),
                "time": time.strftime("%H:%M:%S"),
                "channels": [1, 2, 3, 4, 5, 6],
                "sensor": ["RAW"] * 6,
                "label": ["A1", "A2", "A3", "A4", "A5", "A6"],
                "column": ["nSeq", "I1", "I2", "O1", "O2", "A1", "A2", "A3", "A4", "A5", "A6"],
                "special": [{}, {}, {}, {}, {}, {}],
                "digital IO": [0, 0, 1, 1],
            }
        }

        with open(path, 'w', newline='') as f:
            f.write("# OpenSignals Text File Format. Version 1\n")
            f.write(f"# {json.dumps(header_obj)}\n")
            f.write("# EndOfHeader\n")
            seq = 0
            for _, sample in samples:
                row = [seq, 0, 0, 0, 0]
                # Ignorer le premier élément de sample qui est le nSeq du LSL stream
                # et prendre les 6 canaux analogiques suivants
                analog = list(sample[1:7]) if len(sample) > 1 else list(sample)
                if len(analog) < 6:
                    analog.extend([0] * (6 - len(analog)))
                row.extend(analog[:6])
                # Convertir toutes les valeurs en int pour correspondre au format OpenSignals
                f.write("\t".join(str(int(v)) for v in row) + "\t\n")
                seq = (seq + 1) % 16  # Cycle 0-15
        
        # Traiter automatiquement les données et mettre à jour le dataset
        self._update_dataset_with_session(path, user_id, workout_id)
        
        return os.path.abspath(path)

    def _next_workout_id(self, data_dir, user_id):
        existing = [name for name in os.listdir(data_dir) if name.lower().startswith(f"sport_user{user_id}_workout")]
        max_id = -1
        for name in existing:
            digits = ''.join(ch for ch in name if ch.isdigit())
            try:
                parts = name.lower().split('_')
                for p in parts:
                    if p.startswith('workout'):
                        wid = int(p.replace('workout', '').replace('.txt', ''))
                        max_id = max(max_id, wid)
            except Exception:
                continue
        return max_id + 1

    def _update_dataset_with_session(self, session_path, user_id, workout_id):
        """
        Traite la nouvelle séance et l'ajoute au dataset des features.
        """
        try:
            # Traiter le fichier pour extraire les features
            df_new = process_file(session_path, window_s=20, overlap=0.5, label=1, workout_id=workout_id)
            
            if df_new.empty:
                print(f"Aucune donnée extraite de {session_path}")
                return
            
            # Ajouter les métadonnées
            df_new['activity'] = 'sport'
            df_new['user'] = user_id
            
            # Charger le dataset existant ou créer un nouveau
            features_dir = os.path.join(self.output_dir, 'features')
            os.makedirs(features_dir, exist_ok=True)
            dataset_path = os.path.join(features_dir, 'features_dataset.csv')
            
            if os.path.exists(dataset_path):
                df_existing = pd.read_csv(dataset_path)
                df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_updated = df_new
            
            # Sauvegarder le dataset mis à jour
            df_updated.to_csv(dataset_path, index=False)
            print(f"Dataset mis à jour: {len(df_new)} nouvelles fenêtres ajoutées (total: {len(df_updated)})")
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour du dataset: {e}")
            import traceback
            traceback.print_exc()

    @Slot()
    def update_live_values(self):
        if not self.live_reader:
            return
        latest = self.live_reader.get_latest()
        sample = latest.get('sample')
        ch = int(latest.get('channel_count') or 0)
        if sample is None:
            return
        chan_labels = latest.get('channel_labels') or []
        for i in range(min(ch - 1, 3)):
            if i + 1 < len(sample):
                val = sample[i + 1]
                name = chan_labels[i + 1] if i + 1 < len(chan_labels) and chan_labels[i + 1] else f"Ch{i+1}"
                self.live_labels[i].setText(f"{name}: {val:.3f}")
            else:
                name = chan_labels[i + 1] if i + 1 < len(chan_labels) and chan_labels[i + 1] else f"Ch{i+1}"
                self.live_labels[i].setText(f"{name}: --")
        for i in range(max(0, ch - 1), 3):
            name = chan_labels[i + 1] if i + 1 < len(chan_labels) and chan_labels[i + 1] else f"Ch{i+1}"
            self.live_labels[i].setText(f"{name}: --")
        
        if self.live_reader:
            m = self.live_reader.get_latest_metrics()
            if m.get('hr_bpm') is not None:
                self.live_hr_label.setText(f"HR: {m['hr_bpm']:.1f} bpm")
            else:
                self.live_hr_label.setText("HR: -- bpm")
            if m.get('hrv_rmssd_ms') is not None:
                self.live_hrv_label.setText(f"HRV (RMSSD): {m['hrv_rmssd_ms']:.0f} ms")
            else:
                self.live_hrv_label.setText("HRV (RMSSD): -- ms")
            if m.get('resp_rate_bpm') is not None:
                self.live_resp_label.setText(f"Resp: {m['resp_rate_bpm']:.1f} bpm")
            else:
                self.live_resp_label.setText("Resp: -- bpm")
            if m.get('emg_rms') is not None:
                self.live_emg_label.setText(f"EMG RMS: {m['emg_rms']:.3f}")
            else:
                self.live_emg_label.setText("EMG RMS: --")
    
    def run_timer(self):
        while self.is_running:
            if not self.is_paused:
                self.elapsed_seconds += 1
                self.phase_elapsed += 1
                from PySide6.QtCore import QMetaObject
                QMetaObject.invokeMethod(self, "update_display", Qt.QueuedConnection)
            time.sleep(1)
    
    @Slot()
    def update_display(self):
        minutes = self.elapsed_seconds // 60
        seconds = self.elapsed_seconds % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")
        
        progress = min(self.elapsed_seconds, self.target_duration)
        self.progress_bar.setValue(progress)
        
        percentage = int((self.elapsed_seconds / self.target_duration) * 100) if self.target_duration > 0 else 0
        self.progress_label.setText(f"{percentage}%")

        # Gestion des phases UNIQUEMENT pour l'entraînement initial
        if self.initial_phases:
            # Trouver la phase actuelle
            time_in_session = self.elapsed_seconds
            current_time = 0
            
            for i, phase in enumerate(self.initial_phases):
                phase_duration = phase['duration']
                if current_time + phase_duration > time_in_session:
                    self.current_phase_index = i
                    remaining = phase_duration - (time_in_session - current_time)
                    self.phase_label.setText(f"Phase: {phase['label']}")
                    self.phase_remaining_label.setText(f"Temps restant phase: {remaining // 60:02d}:{remaining % 60:02d}")
                    break
                current_time += phase_duration
        
        # Pas de gestion de phases pour les entraînements normaux
        
        if self.elapsed_seconds >= self.target_duration:
            self.timer_label.setStyleSheet("color: #e74c3c;")
            self.finish_session()
