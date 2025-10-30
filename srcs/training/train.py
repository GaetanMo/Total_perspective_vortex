from training.edf import get_processed_data
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from training.myCSP import MyCSP
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import joblib

def train():
	X, y = get_processed_data() # X tuple of epochs, y array of labels

	csp = MyCSP(n_components=8, log=True)

	print("X shape:", X.shape)
	clf = Pipeline([
		('CSP', csp),
		('LDA', LinearDiscriminantAnalysis())
	]
	)
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
	joblib.dump(clf, "model.joblib")

	print("Scores par fold :", scores)
	print("Précision moyenne :", np.mean(scores))
	print("Model trained successfully.")