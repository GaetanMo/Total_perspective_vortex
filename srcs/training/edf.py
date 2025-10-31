import mne
from mne import Epochs, events_from_annotations
from training.utils import load_eeg_data, get_all_edf_files, get_epochs_by_event
from prediction.utils import load_eeg_file, get_datastream
import numpy as np

def get_processed_data(mode, subject_number=None, run_number=None):
	mne.set_log_level('WARNING') # Reduce verbosity of MNE logs to warnings and errors only

	if mode == "all":
		edf_files = get_all_edf_files()
		raws = load_eeg_data(edf_files)
		epochs_T1, epochs_T2 = get_epochs_by_event(raws)
		y = np.array([0]*epochs_T1.shape[0] + [1]*epochs_T2.shape[0]) # Labels: 0, 0, 0... for T1, 1, 1, 1... for T2
		X = np.concatenate([epochs_T1, epochs_T2], axis=0) # Combine T1 and T2 epochs into a single dataset
	else:
		raw = load_eeg_file(subject_number, run_number)
		epochs_T1, epochs_T2 = get_datastream(raw)
		y = np.array([0]*epochs_T1.shape[0] + [1]*epochs_T2.shape[0]) # Labels: 0, 0, 0... for T1, 1, 1, 1... for T2
		X = np.concatenate([epochs_T1, epochs_T2], axis=0) # Combine T1 and T2 epochs into a single dataset
		return X, y
	return X, y
