from training.edf import get_processed_data
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

def train():
	X, y = get_processed_data() # X tuple of epochs, y array of labels

	csp = CSP(n_components=4, log=True)

	clf = Pipeline([
		('CSP', csp),
		('LDA', LinearDiscriminantAnalysis())
	])
	clf.fit(X, y)

	print("Model trained successfully.")