from training.edf import get_processed_data
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from training.myCSP import MyCSP
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def train(subject_number, run_number):
	mne.set_log_level('WARNING') # Reduce verbosity of MNE logs to warnings and errors only
	if subject_number == 0 and subject_number == 0:
		print("Training on all dataset...")
		try :
			X, y = get_processed_data("all")
		except FileNotFoundError as e:
			print(f"Error: {str(e)}")
			return
		except Exception as e:
			raise e
	else:
		try:
			X, y = get_processed_data("single", subject_number, run_number) # X tuple of epochs, y array of labels
		except Exception as e:
			raise e
	
	csp = MyCSP(n_components=8, log=True)

	clf = Pipeline([
		('CSP', csp),
		('scaler', StandardScaler()), # Normalize features
		('LDA', LinearDiscriminantAnalysis()) # Classifier, LDA is simple and effective for BCI tasks, with linears separations
	])
	clf.fit(X, y)
	joblib.dump(clf, "model.joblib")

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Data shuffle and split in 5 parts, 4 for training, 1 for testing
	scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

	print(scores)
	print("cross_val_score :", np.mean(scores))
	print("Model trained successfully.")