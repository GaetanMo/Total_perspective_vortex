import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def compute_fft(segment, sfreq):
    N = len(segment)
    freqs = np.fft.rfftfreq(N, d=1/sfreq)  # fréquences en Hz
    fft_vals = np.fft.rfft(segment)
    fft_amp = np.abs(fft_vals)
    return freqs, fft_amp


def preprocess():
    input_folder = "/home/gmorel/IA/tpv_github/srcs/data"
    output_folder = "/home/gmorel/IA/tpv_github/srcs/plots"

    os.makedirs(output_folder, exist_ok=True)

    all_t0_segments = []
    all_t1_segments = []
    all_t2_segments = []
    window_duration = 2

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".edf"):
            file_path = os.path.join(input_folder, filename)
            print(f"Traitement de {filename}...")

            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            raw.filter(1, 40, fir_design='firwin')
            channel = "C3.."
            picks = mne.pick_channels(raw.info['ch_names'], include=[channel])
            data = raw.get_data(picks=picks)[0]
            sfreq = raw.info['sfreq']
            annots = raw.annotations
            if not annots:
                print(f"Pas d'annotations dans {filename} – skip.")
                continue
            n_samples = int(window_duration * sfreq)
            half_window = n_samples // 2
        for annot in annots:
            onset_sample = int(annot['onset'] * sfreq)  # Temps en samples
            start = max(0, onset_sample - half_window)
            end = min(len(data), start + n_samples)  # Évite de dépasser la fin
            
            if end - start < n_samples:
                continue  # Skip si segment trop court
            
            segment = data[start:end]
            
            if annot['description'] == 'T0':
                all_t0_segments.append(segment)
            elif annot['description'] == 'T1':
                all_t1_segments.append(segment)
            elif annot['description'] == 'T2':
                all_t2_segments.append(segment)
    t0_array = np.array(all_t0_segments)
    t1_array = np.array(all_t1_segments)
    t2_array = np.array(all_t2_segments)
    np.save(os.path.join(output_folder, 't0_segments.npy'), t0_array)
    np.save(os.path.join(output_folder, 't1_segments.npy'), t1_array)
    np.save(os.path.join(output_folder, 't2_segments.npy'), t2_array)
    print(f"Extraction terminée !")
    print(f"T0 (repos) : {len(t0_array)} segments")
    print(f"T1 : {len(t1_array)} segments")
    print(f"T2 : {len(t2_array)} segments")
    print(f"Sauvegardés dans {output_folder}")
    print("Tous les fichiers ont été traités !")