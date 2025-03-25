# signal processing pipeline

#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os

# Load the sample EEG dataset
# Load EEG data
eeg_eog_data = pd.read_csv(r"C:\Users\carol\Documents\VSPrograms\Signal_Processing\Rehearsal_031322\Subject1\EEG\D1_EEG_EOG.csv", header=None)
#28 channels of EEG data and 4 channels in the bottom are EOG data
#channel names
#remove last 4 rows from eeg_data to only have eeg data
eeg_data = eeg_eog_data.iloc[:-4,:]
ch_names =['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','C3','Cz','C4','CP5','CP1','CP2','CP6',
           'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']

print("EEG-EOG Data Shape:", eeg_eog_data.shape)
print("EEG-EOG Data Shape:", eeg_data.shape)
# Sampling frequency of the EEG data
sfreq = 1000  

# Create MNE info object
info = mne.create_info(ch_names, sfreq, ch_types=['eeg'] * len(ch_names))



# Create the raw array
raw = mne.io.RawArray(eeg_data, info)


# Plot the raw data
#raw.plot(duration=10, n_channels=len(ch_names), scalings='auto')
plt.show(block=True)


# Plot the power spectral density (PSD) of the EEG signals

#Filtering to only keep the frequencies of interest 
# High-pass filter to remove slow drifts (frequencies below 1 Hz)
raw.filter(1., None)

# Low-pass filter to remove high-frequency noise (above 50 Hz)
raw.filter(None, 50.)
raw.plot(duration=10, n_channels=len(ch_names), scalings='auto')
plt.show(block=True)
raw.plot_psd(fmin=1, fmax=50)
plt.show(block=True)

