from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MyCSP(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=4, log=True):
		self.n_components = n_components
		self.log = log
		self.filters_ = None
	
	def cov_normalized(self, X):
		C = np.dot(X, X.T)
		C /= np.trace(C)  # Normalize
		return C

	def fit(self, X, y=None):
		idx_T1 = np.where(y == 0)[0]
		idx_T2 = np.where(y == 1)[0]

		C_T1 = np.mean([self.cov_normalized(X[i]) for i in idx_T1], axis=0)
		C_T2 = np.mean([self.cov_normalized(X[i]) for i in idx_T2], axis=0)

		eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(C_T1 + C_T2).dot(C_T1))
		ix = np.argsort(eigvals)[::-1]
		eigvecs = eigvecs[:, ix]

		self.filters_ = np.hstack([eigvecs[:, :self.n_components], eigvecs[:, -self.n_components:]])
		return self

	def transform(self, X):
		features = []
		for trial in X:
			filtered = self.filters_.T.dot(trial)
			if self.log:
				features.append(np.log(np.var(filtered, axis=0)))
			else:
				features.append(np.var(filtered, axis=0))
		return np.array(features)

