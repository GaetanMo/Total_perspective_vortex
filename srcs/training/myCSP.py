from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MyCSP(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=4, log=True):
		self.n_components = n_components
		self.log = log
		self.filters_ = None
	
	def cov_normalized(self, X):
		C = np.dot(X, X.T) # Covariance matrix (Channels x Channels), we are interested with chanel1_value1 x chanel1_value1, etc...
		# Example with 4 channels:		
		# 1 2 3 4		30 6  5
		# 0 1 0 1   --> 6  2  0
		# 2 0 1 0       5  0  5
		
		# Variance canal 1 = 1^2 + 2^2 + 3^2 + 4^2 = 30 etc...
		# 30 is variance for canal 1, 2 for canal 2, 5 for canal 3
		# 6 is covariance between canal 1 and canal 2, etc...
		C /= np.trace(C)  # Normalize with total variance of canals (30 + 2 + 5 = 37 in the example)
		return C

	def fit(self, X, y=None):
		idx_T1 = np.where(y == 0)[0]
		idx_T2 = np.where(y == 1)[0]

		C_T1 = np.mean([self.cov_normalized(X[i]) for i in idx_T1], axis=0)
		C_T2 = np.mean([self.cov_normalized(X[i]) for i in idx_T2], axis=0)

		# Solve the generalized eigenvalue problem: find directions (spatial filters) that maximize variance for class T1 relative to T2
		eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(C_T1 + C_T2).dot(C_T1))  
		# - C_T1: covariance matrix of class T1
		# - C_T2: covariance matrix of class T2
		# - (C_T1 + C_T2) pseudo-inverse: normalizes by total covariance
		# - dot(C_T1): finds directions where variance for T1 is maximized relative to T2
		# - np.linalg.eig(): returns eigenvalues (eigvals) and eigenvectors (eigvecs)
		#   - eigvals: tell us how discriminative each direction is
		#   - eigvecs: directions (spatial filters) in the channel space

		# Sort the eigenvalues in descending order
		ix = np.argsort(eigvals)[::-1]  
		# - np.argsort(eigvals): gives indices that would sort eigenvalues ascending

		# Reorder the eigenvectors according to sorted eigenvalues
		eigvecs = eigvecs[:, ix]  
		# - Columns of eigvecs now correspond to eigenvectors sorted by importance
		# - First columns: directions with largest variance for T1
		# - Last columns: directions with largest variance for T2

		self.filters_ = np.hstack([eigvecs[:, :self.n_components], eigvecs[:, -self.n_components:]])
		# Get filter matrix by selecting n_components from both ends 
		return self

	def transform(self, X):
		features = []
		for epoch in X: # For each epoch
			filtered = self.filters_.T.dot(epoch)
			features.append(np.log(np.var(filtered, axis=0)))
		return np.array(features) # Return features as array (n_epochs, n_features (2*n_components))
	