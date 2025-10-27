import numpy as np
import matplotlib.pyplot as plt


def fourier_transform(raw_part):
	data, times = raw_part[:, :]
	sfreq = raw_part.info['sfreq']
	n_channels = data.shape[0]
	frequencies = np.fft.fftfreq(len(times), 1/sfreq)

	fft_result = np.fft.fft(data[0, :])
	amplitude = np.abs(fft_result)  # Amplitude de la FFT
	positive_freqs = frequencies[:len(frequencies)//2]
	positive_amplitude = amplitude[:len(amplitude)//2]
	peak_freq = positive_freqs[np.argmax(positive_amplitude)]
	print(f"Peak frequency for channel 0: {peak_freq} Hz")


	plt.plot(frequencies[:len(frequencies)//2], amplitude[:len(amplitude)//2])
	plt.title('Fourier Transform - Channel 0')
	plt.show()
