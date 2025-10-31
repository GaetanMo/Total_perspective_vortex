import joblib
import mne
import numpy as np
from mne import Epochs, events_from_annotations
from training.utils import apply_filter

def load_eeg_file(subject_number, run_number):
	file_path = f"data/S{subject_number:03d}/S{subject_number:03d}R{run_number:02d}.edf"
	print(f"Loading EEG file: {file_path}")
	try:
		raw = mne.io.read_raw_edf(file_path, preload=True)
		raw = apply_filter(raw)
	except Exception as e:
		print(f"Error while loading {file_path}: {e}, skipping this file.")
		return None
	return raw

def get_datastream(raw):
	epochs_T1 = None
	epochs_T2 = None

	try :
		events, _ = mne.events_from_annotations(raw, event_id={'T0': 1, 'T1': 2, 'T2': 3}) # Returns events (shape [n_events, 3]) --> for each event gives [start moment (division by sfreq to get data in seconds), 0, event_id]
		Epochs = mne.Epochs(raw, events, event_id={'T1': 2, 'T2': 3}, tmin=-0.2, tmax=3.8, baseline=(None, 0), preload=False, reject=None) # Create epochs for T1 and T2, shape (n_epochs, n_channels, n_times) like raw
		epochs_T1 = Epochs['T1'].get_data() 
		epochs_T2 = Epochs['T2'].get_data()
	except Exception as e:
		pass

	# We need to concatenate them into single arrays
	return epochs_T1, epochs_T2

def load_model():
	import joblib
	try:
		model = joblib.load("model.joblib")
		print("Model loaded successfully.")
		return model
	except Exception as e:
		print(f"Error loading model: {e}")
		return None