import biosignalsnotebooks as bsnb
import os
import matplotlib.pyplot as plt
from numpy import linspace

def load_txt_files(folder_path: str):
    """
    Charge les fichiers txt OpenSignals depuis un dossier donné.
    """

    if not os.path.isdir(folder_path):
        raise ValueError(f"Dossier introuvable: {folder_path}")

    results = []

    for name in os.listdir(folder_path):
        if not name.lower().endswith(".txt"):
            continue
        fpath = os.path.join(folder_path, name)
        if not os.path.isfile(fpath):
            continue

        data, header = bsnb.load(fpath, get_header=True)
        results.append({"data": data, "header": header})

        #collect signals
        EMG = data['CH1']
        ECG = data['CH2']

        #normalize signals
        EMG = (EMG - EMG.mean()) / EMG.std()
        ECG = (ECG - ECG.mean()) / ECG.std()

    return EMG


#f = load_txt_files(r"C:/Users/inesr/Documents/OpenSignals (r)evolution/files/ai_adapt")
#plt.plot(f)
#plt.show()

data, header = bsnb.load("C:/Users/inesr/Documents/OpenSignals (r)evolution/files/ai_adapt/opensignals_002106be1615_2025-12-26_18-09-55.txt", get_header=True)
signal = data['CH1']
# Sampling rate and acquired data
sr = header["sampling rate"]
time = linspace(0, len(signal) / sr, len(signal))
activation_begin, activation_end = bsnb.detect_emg_activations(signal, sr)[:2]
bsnb.plot_compare_act_config(signal, sr, file_name="C:/Users/inesr/OneDrive/Documents/cours/M2/Ai_ADAPT/projet_rain/other/test.png",)