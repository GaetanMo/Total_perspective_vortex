import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from utils import load_eeg_data
from transform import fourier_transform
from utils import cut_eeg_data_in_2s_blocks

def preprocess_eeg():
    raw = load_eeg_data('/home/gmorel/IA/tpv_github/srcs/data/S001R03.edf')
    raw.filter(l_freq=8, h_freq=30)
    cuted_raw = cut_eeg_data_in_2s_blocks(raw)
    print(f"Number of 2s segments: {len(cuted_raw)}")
    print(cuted_raw[7].annotations)
    # fourier_transform(raw)
    