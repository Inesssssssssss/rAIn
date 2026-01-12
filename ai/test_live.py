from pylsl import StreamInlet, resolve_byprop
from pylsl import resolve_streams

streams = resolve_streams()

print(f"{len(streams)} stream(s) trouvé(s)\n")

for s in streams:
    print("Nom :", s.name())
    print("Type :", s.type())
    print("Source ID :", s.source_id())
    print("Canaux :", s.channel_count())
    print("Fréquence :", s.nominal_srate())
    print("-" * 30)

# Chercher les streams disponibles
print("Recherche de streams BITalino...")
streams = resolve_byprop('name', 'OpenSignals')  # ou 'Biosignals'
print(f"{len(streams)} stream(s) BITalino trouvé(s)")

# Connexion au premier stream trouvé
inlet = StreamInlet(streams[0])

# Réception données temps réel
while True:
    sample, timestamp = inlet.pull_sample()
    ecg_value = sample[0]  # Premier canal
    
    # Traiter en temps réel
    print(f"ECG: {ecg_value} à {timestamp}")