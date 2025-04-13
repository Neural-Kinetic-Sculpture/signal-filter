# signal processing pipeline

#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
from pylsl import StreamInlet, resolve_byprop

# H-infinity filter class
class HInfinityFilter:
    def __init__(self, ref_dim, gamma=1.15, q=1e-10, p0=0.5):
        self.gamma = gamma
        self.q = q
        self.ref_dim = ref_dim
        self.P = p0 * np.eye(ref_dim)
        self.w = np.zeros((ref_dim, 1))

    def update(self, s, r):
        r = r.reshape(-1, 1)
        y = s - (self.w.T @ r).item()
        denominator = self.gamma**2 + (r.T @ self.P @ r).item()
        self.P = self.P - (self.P @ r @ r.T @ self.P) / denominator
        self.w = self.w + self.P @ r * y
        return float(y)

# Configuration
BUFFER_SIZE = 500  # Number of samples per batch
SFREQ = 1000  # Sampling rate (update based on LiveAmp settings)
LOW_CUTOFF = 1  # High-pass filter cutoff (Hz)
HIGH_CUTOFF = 50  # Low-pass filter cutoff (Hz)

def connect_to_stream():
    #Try to connect to the LiveAmp EEG stream.
    print("Searching for LiveAmp EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=5) 
    return StreamInlet(streams[0]) if streams else None

# Initial connection
inlet = connect_to_stream()
if inlet is None:
    print("⚠️ No EEG stream found. Please start the LiveAmp stream and restart the script.")
    exit()

print("✅ Connected to LiveAmp EEG stream!")


# Load the sample EEG dataset
# Load EEG data
current_dir = os.getcwd()
eeg_eog_data = pd.read_csv(r"C:\Users\carol\Documents\VSPrograms\Signal_Processing\Rehearsal_031322\Subject1\EEG\D1_EEG_EOG.csv", header=None)
#28 channels of EEG data and 4 channels in the bottom are EOG data
#channel names
#remove last 4 rows from eeg_data to only have eeg data
eeg_data = eeg_eog_data.iloc[:-4,:]
ch_names =['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','C3','Cz','C4','CP5','CP1','CP2','CP6',
           'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']

eog_ch_names = ['EOG1', 'EOG2', 'EOG3', 'EOG4']

#trying to see eeg with the eog data
eeg_eog_data_ch_names = ['EEG_EOG'] * 32
raw_ch_types = ['eeg'] * len(eeg_ch_names) + ['eog'] * len(eog_ch_names)
eeg_eog_info = mne.create_info(eeg_eog_data_ch_names, 1000, ch_types=raw_ch_types)

raw = mne.io.RawArray(eeg_eog_data, eeg_eog_info)
raw.plot(duration=10, n_channels=32, scalings='auto')
plt.title('Raw Data Plot')
plt.show(block=True)
raw.plot_psd(fmin=1, fmax=50)
plt.title('Raw PSD Plot')
plt.show(block=True)
eeg_data = eeg_data.values
eog_data = eog_data.values
#Insert H infinity filter here 
# Step 1: Construct EOG-based reference signals
VEOG = eog_data[0, :] - eog_data[1, :]  # Vertical eye movement
HEOG = eog_data[3, :] - eog_data[2, :]  # Horizontal eye movement
ones = np.ones(VEOG.shape)              # Drift/bias component
# Step 2: Combine reference signals
ref_signals = np.vstack((VEOG, HEOG, ones))  # Shape: (3, N)

# Step 3: Prepare output array
clean_eeg = np.zeros_like(eeg_data)

# Step 4: Apply H-infinity filter per channel
for ch in range(eeg_data.shape[0]):
    hinf = HInfinityFilter(ref_dim=3, gamma=1.15, q=1e-10, p0=0.5)
    for t in range(eeg_data.shape[1]):
        s = eeg_data[ch, t]        # Noisy EEG
        r = ref_signals[:, t]      # Reference EOG + drift
        clean_eeg[ch, t] = hinf.update(s, r)  # Cleaned signal


# Create MNE Info for just the EEG channels
info = mne.create_info(ch_names=eeg_ch_names, sfreq=1000, ch_types='eeg')

# Create RawArray from cleaned EEG
raw_clean = mne.io.RawArray(clean_eeg, info)


raw_clean.plot(duration=10, n_channels=len(eeg_ch_names), scalings='auto')
plt.title('Data after H-infinity filter Plot')
plt.show(block=True)
raw_clean.plot_psd(fmin=1, fmax=50)
plt.title('Data after H-infinity filter PSD Plot')
plt.show(block=True)

#High pass filter to remove slow drifts (frequencies below 1 Hz)
raw_clean.filter(LOW_CUTOFF, None)

# Low-pass filter to remove high-frequency noise (above 50 Hz)
raw_clean.filter(None, HIGH_CUTOFF)

raw_clean.plot(duration=10, n_channels=len(eeg_ch_names), scalings='auto')
plt.title('Data after High and Low filters Plot')
plt.show(block=True)
raw_clean.plot_psd(fmin=1, fmax=50)
plt.title('Data after High and Low filters PSD Plot')
plt.show(block=True)


#robust referencing
raw_clean.set_eeg_reference(ref_channels='average', projection=True)
raw_clean.apply_proj()


raw_clean.plot(duration=10, n_channels=len(eeg_ch_names), scalings='auto')
plt.title('Clean Data Plot')
plt.show(block=True)
raw_clean.plot_psd(fmin=1, fmax=50)
plt.title('Clean PSD Plot')
plt.show(block=True)
