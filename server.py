
#==============================================================
#Signal Processing Pipeline for EEG Data
#==============================================================


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import time
from collections import deque
import os
from scipy.signal import welch
from pylsl import StreamInlet, resolve_byprop

from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import random
import threading
import os
import json

# Global variables 
average_dominant_freq = -1
psd = -1

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
EEG_CHANNELS = 4      # First 4 channels = EEG
EOG_CHANNELS = 4       # Last 4 channels = EOG


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



            # Output processed batch timestamp
            print(f"Processed batch at {time.strftime('%H:%M:%S')}")

except KeyboardInterrupt:
    print("\nReal-time EEG filtering stopped.")
########

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", json=json)
CORS(app)

clients_connected = 0
thread = None

def get_eeg_data():
    # # Simulate frequency drift in alpha range
    # dominant_freq = round(random.gauss(10.3, 0.6), 2)  # ~10.3 Hz ± 0.6
    # dominant_freq = max(8.0, min(dominant_freq, 12.0))  # Clamp to alpha band

    # # Simulate PSD (signal power), e.g. 0-100
    # psd = round(random.uniform(0.0, 100.0), 2)

    # Confidence is ratio of alpha power to total brain power
    confidence = round(random.uniform(0.7, 0.95), 2)

    eeg_data = {
        "wave_type": "None", #ignore
        "dominant_freq": average_dominant_freq,
        "psd": psd,
        "confidence": confidence, #ignore
        "timestamp": round(time.time(), 3)  # float seconds with millisecond precision
    }

    return eeg_data

@app.route('/')
def index():
    return "🎛️ Neural Kinetic Sculpture EEG WebSocket Server Running"

@socketio.on('connect')
def handle_connect():
    global thread, clients_connected
    clients_connected += 1
    print(f'Client connected: {clients_connected}')
    socketio.emit('test_response', '✅ Connected to EEG WebSocket server')
    
    if thread is None:
        thread = socketio.start_background_task(send_eeg_data)

@socketio.on('disconnect')
def handle_disconnect():
    global clients_connected
    clients_connected = max(0, clients_connected - 1)
    print(f'Client disconnected. Remaining: {clients_connected}')

@socketio.on('test_connection')
def handle_test(message):
    print(f"Test connection message received: {message}")
    socketio.emit('test_response', '✅ Echo from server')

def send_eeg_data():
    """Background task to emit EEG data every 2 seconds"""
    while True:
        socketio.sleep(2)  # More stable and realistic pacing
        if clients_connected > 0:
            eeg_data = get_eeg_data()
            print("📡 Emitting EEG Data:", eeg_data)
            socketio.emit('eeg_data', json.dumps(eeg_data))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)

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
