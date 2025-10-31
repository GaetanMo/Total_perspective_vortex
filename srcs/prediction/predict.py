from prediction.utils import load_model, get_datastream, load_eeg_file
import numpy as np
import mne

def predict(subject_number, run_number):

	model = load_model()
	if model is None:
		print("No model available for prediction.")
		return

	raw = load_eeg_file(subject_number, run_number)
	epochs_T1, epochs_T2 = get_datastream(raw)

	y = np.array([0]*epochs_T1.shape[0] + [1]*epochs_T2.shape[0]) # Labels: 0, 0, 0... for T1, 1, 1, 1... for T2
	X = np.concatenate([epochs_T1, epochs_T2], axis=0) # Combine T1 and T2 epochs into a single dataset

	predictions = model.predict(X)
	accuracy = np.mean(predictions == y)
	print(f"Prediction accuracy on test data: {accuracy*100:.2f}%")
	# Here load new EEG data real stream simulation