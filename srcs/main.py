from training.train import train
from prediction.predict import predict
import argparse

def main():
	parser = argparse.ArgumentParser(description="Scrpt to train or predict EEG data.")

	parser.add_argument('param1', type=int, help='Subject number (1 to 109)')
	parser.add_argument('param2', type=int, help='Run number (1 to 14)')
	parser.add_argument('action', choices=['train', 'predict'], help='Mode train or predict')
	args = parser.parse_args()

	if args.param1 == 0 and args.param2 == 0 and args.action == 'train':
		train(0, 0)
		return
	elif args.param1 < 1 or args.param1 > 109:
		print("Subject number must be between 1 and 109.")
		return
	elif args.param2 < 1 or args.param2 > 14:
		print("Run number must be between 1 and 14.")
		return
	elif args.action == 'train':
		train(args.param1, args.param2)
	elif args.action == 'predict':
		predict(args.param1, args.param2)


if __name__ == "__main__":
	main()