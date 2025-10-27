import mne

def cut_eeg_data_in_2s_blocks(raw):
	sfreq = raw.info['sfreq']
	segment_duration = 2

	# Calculer le nombre d'échantillons par segment de 2 secondes
	n_samples_per_segment = int(sfreq * segment_duration)
	n_samples_total = raw.n_times
	# Découper les données en blocs de 2 secondes
	segments = []
	annotations = raw.annotations
	for start_sample in range(0, n_samples_total, n_samples_per_segment):
		
		# Prendre un segment de 2 secondes (n_samples_per_segment échantillons)
		end_sample = start_sample + n_samples_per_segment
		if end_sample > n_samples_total:
			end_sample = n_samples_total
			end_sample -= 1
		segment_data = raw.copy().crop(tmin=start_sample / sfreq, tmax=end_sample / sfreq)
		segment_annotations = annotations.copy().crop(tmin=start_sample / sfreq, tmax=end_sample / sfreq)
		if len(segment_annotations) > 0:
			last_annotation = segment_annotations[-1]  # La dernière annotation
			segment_data.set_annotations(mne.annotations.Annotations(
				onset=[last_annotation.onset],
				duration=[last_annotation.duration],
				description=[last_annotation.description]
		))
		segments.append(segment_data)
	return segments

def load_eeg_data(file_path):
	raw = mne.io.read_raw_edf(file_path, preload=True)
	return raw