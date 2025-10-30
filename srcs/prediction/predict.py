import joblib

def predict():
	
	try :
		model = joblib.load("model.joblib")
		print("Model loaded successfully.")
	except Exception as e:
		print(f"Error loading model: {e}")
		return
	# Here load new EEG data real stream simulation