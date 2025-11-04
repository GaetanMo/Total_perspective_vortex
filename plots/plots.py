
import mne
import matplotlib.pyplot as plt

fichier_mne = 'data/S001/S001R05.edf'
raw = mne.io.read_raw_edf(fichier_mne, preload=True)

fig = raw.plot(duration=10, n_channels=4, scalings='auto', show=False)
fig.savefig('plots/file_without_filter.png', dpi=300)

raw.filter(8, 30)
fig = raw.plot(duration=10, n_channels=4, scalings='auto', show=False)
fig.savefig('plots/file_with_filter.png', dpi=300)