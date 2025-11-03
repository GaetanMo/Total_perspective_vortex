from training.train import train
from prediction.predict import predict

def default():
	train(0, 0)
	mean_accuracy = 0
	k = 0
	for i in range(1, 110):
		for j in range(3, 15):
			try :
				accuracy = predict(i, j, "default")
				print(f"Subject {i:02d}, Run {j:02d}, Accuracy: {accuracy:.2f}%")
				mean_accuracy += accuracy
				k += 1
			except Exception as e:
				continue
	mean_accuracy /= k
	print(f"Mean Accuracy over all subjects and runs: {mean_accuracy:.2f}%")
	

	
