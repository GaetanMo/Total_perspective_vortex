
import mne
import matplotlib.pyplot as plt

mne_file = 'data/S001/S001R05.edf'
mne.set_log_level('WARNING')
raw = mne.io.read_raw_edf(mne_file, preload=True)

fig = raw.plot(duration=10, n_channels=4, scalings='auto', show=False)
fig.savefig('plots/file_without_filter.png', dpi=300)

raw.filter(8, 40)
fig = raw.plot(duration=10, n_channels=4, scalings='auto', show=False)
fig.savefig('plots/file_with_filter.png', dpi=300)