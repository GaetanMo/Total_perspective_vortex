from pathlib import Path
import mne
import numpy as np

def get_epochs_by_event(raws):
	epochs_T1 = []
	epochs_T2 = []

	for raw in raws:
		try :
			events, _ = mne.events_from_annotations(raw, event_id={'T0': 1, 'T1': 2, 'T2': 3}) # Returns events (shape [n_events, 3]) --> for each event gives [start moment (division by sfreq to get data in seconds), 0, event_id]
			Epochs = mne.Epochs(raw, events, event_id={'T1': 2, 'T2': 3}, tmin=-0.2, tmax=3.8, baseline=(None, 0), preload=False, reject=None) # Create epochs for T1 and T2, shape (n_epochs, n_channels, n_times) like raw
			epochs_T1.append(Epochs['T1'].get_data()) # (n_epochs, n_channels, n_times) # For each epoch, gives the data for each channel at each time point, same structure as raw.get_data(), but only for the selected epochs
			epochs_T2.append(Epochs['T2'].get_data())
		except Exception as e:
			pass

	# epochs_T1 and epochs_T2 are lists of arrays, each array corresponding to one raw file
	# We need to concatenate them into single arrays

	X_T1 = np.concatenate(epochs_T1, axis=0) # Concatenate all epochs along the first axis (n_epochs) to get a single array (total_epochs_T1, n_channels, n_times)
	X_T2 = np.concatenate(epochs_T2, axis=0)
	return X_T1, X_T2

def apply_filter(raw):
	raw.filter(l_freq=8, h_freq=40)
	return raw

def load_eeg_data(file_paths):
	raws = []
	for file_path in file_paths:
		try:
			raw = mne.io.read_raw_edf(file_path, preload=True)
			raws.append(apply_filter(raw))
		except Exception as e:
			print(f"Error while loading {file_path}: {e}, skipping this file.")
			raise e
	return raws

def get_all_edf_files():
	script_dir = Path(__file__).parent
	root_dir = script_dir.parents[1] / "data"
	edf_files = list(root_dir.rglob("*.edf"))
	if not edf_files:
		raise FileNotFoundError("No .edf files found in the specified directory.")
	return edf_files
