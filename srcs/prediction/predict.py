from prediction.utils import load_model, get_datastream, load_eeg_file
import numpy as np
import mne
import time

def predict(subject_number, run_number, mode="single"):
	mne.set_log_level('WARNING') # Reduce verbosity of MNE logs to warnings and errors only
	if subject_number < 1 or subject_number > 109:
		print("Subject number must be between 1 and 109.")
		return
	
	model = load_model()
	if model is None:
		print("No model available for prediction.")
		return

	try :
		raw = load_eeg_file(subject_number, run_number)
		epochs_in_order, labels = get_datastream(raw)
	except Exception as e:
		raise e
	if mode == "single":
		print("epoch nb: [prediction] [truth] equal?")
	# Simulating data stream prediction
	# epoch = []
	accuracy = 0
	for i in range(len(epochs_in_order)):
		# time.sleep(8.2)  # Simulate real-time delay
		current_epoch = epochs_in_order[i][np.newaxis, :, :]
		# epoch.append(epochs_in_order[i])
		prediction = model.predict(current_epoch)
		if prediction[0] == labels[i]:
			accuracy += 1
		display_pred = 1 if prediction[0] == 0 else 2
		display_label = 1 if labels[i] == 0 else 2
		if mode == "single":
			print(f"epoch {i:02d}:      [{display_pred}]      [{display_label}]    {prediction[0] == labels[i]}")
	accuracy = (accuracy / len(labels)) * 100
	if mode == "single":
		print(f"Accuracy: {accuracy:.2f}%")
	return accuracy
	# Here load new EEG data real stream simulation