#==============================================================
#Signal Processing Pipeline for EEG Data
#==============================================================

#Import necessary libraries
import numpy as np
import mne
import time
import requests
from collections import deque
from scipy.signal import welch
from pylsl import StreamInlet, resolve_byprop
import os

import matplotlib.pyplot as plt
import pandas as pd

# Configuration
BUFFER_SIZE = 500  # Number of samples per batch
SFREQ = 1000  # Sampling rate (update based on LiveAmp settings)
LOW_CUTOFF = 1  # High-pass filter cutoff (Hz)
HIGH_CUTOFF = 50  # Low-pass filter cutoff (Hz)
EEG_CHANNELS = 4      # First 4 channels = EEG
EOG_CHANNELS = 4       # Last 4 channels = EOG

# Render server settings
RENDER_SERVER_URL = "https://signal-filter.onrender.com"
API_KEY = os.environ['EEG_API_KEY']

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

def connect_to_stream():
    #Try to connect to the LiveAmp EEG stream.
    print("Searching for LiveAmp EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=5) 
    return StreamInlet(streams[0]) if streams else None

def send_data_to_render(data):
    try:
        response = requests.post(
            RENDER_SERVER_URL,
            json=data,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": API_KEY
            },
            timeout=5 # 5 second timeout
        )
        if response.status_code == 200:
            print(f"✅ Data sent to cloud server: {data['dominant_freq']:.2f} Hz")
        else:
            print(f"⚠️ Error sending data to cloud server: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to send data: {e}")

def classify_wave(freq):
    if 0.5 <= freq < 4:
        return 'delta'
    elif 4 <= freq < 8:
        return 'theta'
    elif 8 <= freq < 12:
        return 'alpha'
    elif 12 <= freq < 30:
        return 'beta'
    elif 30 <= freq <= 50:
        return 'gamma'
    else:
        return 'unknown'

# Initial connection
inlet = connect_to_stream()
if inlet is None:
    print("⚠️ No EEG stream found. Please start the LiveAmp stream and restart the script.")
    exit()
print("✅ Connected to LiveAmp EEG stream!")

# MNE Info setup
eeg_ch_names = ['Fp1','Fp2','F3','F4']
eog_ch_names = ['EOG1', 'EOG2', 'EOG3', 'EOG4']
ch_names = eeg_ch_names + eog_ch_names
ch_types = ['eeg'] * EEG_CHANNELS + ['eog'] * EOG_CHANNELS
info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)

# Create buffer
eeg_buffer = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

# Start real-time processing
print("Starting real-time EEG filtering... (Press Ctrl+C to stop)")
#######
try:
    while True:  # Infinite loop until manually stopped
        sample, timestamp = inlet.pull_sample(timeout=1.0)

        if sample is not None:
            eeg_buffer.append(sample)
            timestamps.append(timestamp)
        else:
            print("⚠️ No data received! Checking connection...")
            time.sleep(2)  # Wait before retrying
            inlet = connect_to_stream()  # Attempt reconnection
            if inlet is None:
                print("❌ LiveAmp EEG stream lost. Waiting for reconnection...")
                time.sleep(5)  # Wait before retrying again
                continue

        # Process when buffer is full
        if len(eeg_buffer) == BUFFER_SIZE:
            data = np.array(eeg_buffer).T  # Shape (channels, samples)
            
            eeg_data = data[:EEG_CHANNELS, :]
            eog_data = data[-EOG_CHANNELS:, :]

            #H-infinity Filtering
            #Insert here
            # Step 1: Construct EOG-based reference signals
            VEOG = eog_data[0, :] - eog_data[1, :]  # Vertical eye movement
            HEOG = eog_data[3, :] - eog_data[2, :]  # Horizontal eye movement
            drift = np.ones(VEOG.shape)              # Drift/bias component
            # Step 2: Combine reference signals
            ref_signals = np.vstack((VEOG, HEOG, drift))  # Shape: (3, N)

            # Step 3: Prepare output array
            clean_eeg = np.zeros_like(eeg_data)

            # Step 4: Apply H-infinity filter per channel
            for ch in range(EEG_CHANNELS):
                hinf = HInfinityFilter(ref_dim=3, gamma=1.15, q=1e-10, p0=0.5)
                for t in range(eeg_data.shape[1]):
                    s = eeg_data[ch, t]        # Noisy EEG
                    r = ref_signals[:, t]      # Reference EOG + drift
                    clean_eeg[ch, t] = hinf.update(s, r)  # Cleaned signal


            # Create MNE Info for just the EEG channels
            info = mne.create_info(ch_names=eeg_ch_names, sfreq=1000, ch_types='eeg')

            # Create RawArray from cleaned EEG
            raw_clean = mne.io.RawArray(clean_eeg, info)

            # Apply real-time filtering
            raw_clean.filter(LOW_CUTOFF, None) # High-pass filter
            raw_clean.filter(None, HIGH_CUTOFF) # Low-pass filter

            #Robust referencing
            raw_clean.set_eeg_reference(ref_channels='average', projection=True)
            raw_clean.apply_proj()

            #Get the average dominant frequency
            # Step 1: Get filtered EEG data (numpy array of shape [n_channels, n_samples])
            eeg_data_filtered = raw_clean.get_data()

            # Step 2: Set PSD parameters
            sfreq = raw_clean.info['sfreq']
            nperseg = 256  # Number of samples per segment for FFT

            # Step 3: Compute dominant frequency for each channel
            dominant_freqs = []
            channel_psds = []

            for ch_data in eeg_data_filtered:
                freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
                valid_band = (freqs >= 1) & (freqs <= 50)
                freqs = freqs[valid_band]
                psd = psd[valid_band]

                # Normalize PSD to range 0-100
                psd_min = np.min(psd)
                psd_max = np.max(psd)
                psd_norm = 100 * (psd - psd_min) / (psd_max - psd_min + 1e-10)  # avoid division by 0

                # Get dominant frequency from normalized PSD (from 0 to 100)
                dom_freq = freqs[np.argmax(psd_norm)]
                dominant_freqs.append(dom_freq)
                channel_psds.append(np.max(psd_norm))  # Store normalized PSD

            average_dominant_freq = np.mean(dominant_freqs)
            average_psd = np.mean(channel_psds)
            print(f"Average dominant frequency: {average_dominant_freq:.2f} Hz")

            # Find wave type
            wave_type = classify_wave(average_dominant_freq)

            # Prepare data for sending to Render server
            eeg_data_to_send = {
                "wave_type": wave_type,
                "dominant_freq": float(average_dominant_freq),
                "psd": float(average_psd),
                "timestamp": time.time()
            }
            # Send the processed data to the Render server
            send_data_to_render(eeg_data_to_send)

            # Output processed batch timestamp
            print(f"Processed batch at {time.strftime('%H:%M:%S')}")

except KeyboardInterrupt:
    print("\nReal-time EEG filtering stopped.")
########

"""
# 
# Load EEG data
#eeg_eog_data = pd.read_csv(r"C: ##INSERT YOUR PATH HERE##", header=None)
#28 channels of EEG data and 4 channels in the bottom are EOG data
#channel names
#remove last 4 rows from eeg_data to only have eeg data
eeg_data = eeg_eog_data.iloc[:-4, :]
eog_data = eeg_eog_data.iloc[-4:, :]
eeg_ch_names =['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','C3','Cz','C4','CP5','CP1','CP2','CP6',
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

# Step 1: Get filtered EEG data (numpy array of shape [n_channels, n_samples])
eeg_data_filtered = raw_clean.get_data()

# Step 2: Set PSD parameters
sfreq = raw_clean.info['sfreq']
nperseg = 256  # Number of samples per segment for FFT

# Step 3: Compute dominant frequency for each channel
dominant_freqs = []

for ch_data in eeg_data_filtered:
    freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
    valid_band = (freqs >= 1) & (freqs <= 50)
    freqs = freqs[valid_band]
    psd = psd[valid_band]
    # Get the frequency with the max power
    dom_freq = freqs[np.argmax(psd)]
    dominant_freqs.append(dom_freq)


average_dominant_freq = np.mean(dominant_freqs)

print(f"Average dominant frequency: {average_dominant_freq:.2f} Hz")


"""
