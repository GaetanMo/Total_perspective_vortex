import joblib
import mne
import numpy as np
from mne import Epochs, events_from_annotations
from training.utils import apply_filter

def load_eeg_file(subject_number, run_number):
	file_path = f"data/S{subject_number:03d}/S{subject_number:03d}R{run_number:02d}.edf"
	try:
		raw = mne.io.read_raw_edf(file_path, preload=True)
		raw = apply_filter(raw)
	except Exception as e:
		raise e
	return raw

def get_datastream(raw):
	epochs_in_order = []

	try :
		events, _ = mne.events_from_annotations(raw, event_id={'T0': 1, 'T1': 2, 'T2': 3}) # Returns events (shape [n_events, 3]) --> for each event gives [start moment (division by sfreq to get data in seconds), 0, event_id]
		epochs = mne.Epochs(raw, events, event_id={'T1': 2, 'T2': 3}, tmin=-0.2, tmax=3.8, baseline=(None, 0), preload=False, reject=None) # Create epochs for T1 and T2, shape (n_epochs, n_channels, n_times) like raw
		epochs_in_order = epochs.get_data()
	except Exception as e:
		raise e
	labels_in_order = epochs.events[:, -1]  # Extract labels from events
	labels_in_order = np.where(labels_in_order == 2, 0, 1) # Remapping labels 0 / 1
	return epochs_in_order, labels_in_order

def load_model():
	import joblib
	try:
		model = joblib.load("model.joblib")
		return model
	except Exception as e:
		return None