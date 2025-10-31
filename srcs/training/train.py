from training.edf import get_processed_data
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from training.myCSP import MyCSP
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import joblib

def train(subject_number, run_number):
	X, y = get_processed_data() # X tuple of epochs, y array of labels

	csp = MyCSP(n_components=8, log=True)

	clf = Pipeline([
		('CSP', csp),
		('LDA', LinearDiscriminantAnalysis())
	]
	)
	clf.fit(X, y)
	joblib.dump(clf, "model.joblib")
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

	print("Scores par fold :", scores)
	print("Précision moyenne :", np.mean(scores))
	print("Model trained successfully.")