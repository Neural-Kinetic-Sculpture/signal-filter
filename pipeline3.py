#Import necessary libraries
import numpy as np
import mne
import time
import requests
from collections import deque
from scipy.signal import welch
from pylsl import StreamInlet, resolve_byprop
import os
from numpy.linalg import inv as Inv

import matplotlib.pyplot as plt
import pandas as pd

# Constants
LOW_CUTOFF = 1.0  # Hz
HIGH_CUTOFF = 50.0  # Hz
CHUNK_SIZE = 1000  # Samples per chunk
SAMPLING_RATE = 1000  # Hz
BUFFER_SIZE = 5  # 5 chunks = 5 seconds of data

# Render server settings
RENDER_SERVER_URL = "https://signal-filter.onrender.com/receive_eeg_data"
API_KEY = 'ZMn6VC5ZuSpFrLdbPYvgf9QW4TskDhjX2EeRK7UJ'

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
    
def calculate_band_powers(freqs, psd):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 50)
    }
    band_powers = {}
    total_power = np.trapz(psd, freqs)
    
    for band, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.trapz(psd[mask], freqs[mask])
        band_powers[band] = band_power / total_power  # Relative power
        
    return band_powers

def process_eeg_chunk(eeg_chunk, eog_chunk, eeg_ch_names, buffer):
    """Process a chunk of EEG and EOG data."""
    # Step 1: Construct EOG-based reference signals
    VEOG = eog_chunk[0, :] - eog_chunk[1, :]  # Vertical eye movement
    HEOG = eog_chunk[3, :] - eog_chunk[2, :]  # Horizontal eye movement
    ones = np.ones(VEOG.shape)              # Drift/bias component
    
    # Step 2: Combine reference signals
    ref_signals = np.vstack((VEOG, HEOG, ones))  # Shape: (3, chunk_size)

    # Step 3: Prepare output array
    clean_eeg = np.zeros_like(eeg_chunk)

    # Step 4: Apply H-infinity filter per channel
    for ch in range(eeg_chunk.shape[0]):
        hinf = HInfinityFilter(ref_dim=3, gamma=1.15, q=1e-10, p0=0.5)
        for t in range(eeg_chunk.shape[1]):
            s = eeg_chunk[ch, t]        # Noisy EEG
            r = ref_signals[:, t]       # Reference EOG + drift
            clean_eeg[ch, t] = hinf.update(s, r)  # Cleaned signal

    # Create MNE Info for just the EEG channels
    info = mne.create_info(ch_names=eeg_ch_names, sfreq=SAMPLING_RATE, ch_types='eeg')

    # Create RawArray from cleaned EEG
    raw_clean = mne.io.RawArray(clean_eeg, info)

    # High pass filter to remove slow drifts (frequencies below 1 Hz)
    raw_clean.filter(LOW_CUTOFF, None)

    # Low-pass filter to remove high-frequency noise (above 50 Hz)
    raw_clean.filter(None, HIGH_CUTOFF)

    # Apply average reference
    raw_clean.set_eeg_reference(ref_channels='average', projection=True)
    raw_clean.apply_proj()

    # Get filtered EEG data
    eeg_data_filtered = raw_clean.get_data()

    # Compute PSD and dominant frequency
    sfreq = raw_clean.info['sfreq']
    nperseg = min(256, eeg_chunk.shape[1])  # Adjust segment size for short chunks
    
    dominant_freqs = []
    all_psds = []
    all_freqs = []
    normalized_psds = []

    for ch_data in eeg_data_filtered:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
        valid_band = (freqs >= 1) & (freqs <= 50)
        freqs = freqs[valid_band]
        psd = psd[valid_band]
        # Get the frequency with the max power
        if len(psd) > 0:
            # Store the full PSDs for later normalization
            all_psds.append(psd)
            all_freqs.append(freqs)
            
            # Get the frequency with the max power
            dom_freq = freqs[np.argmax(psd)]
            dominant_freqs.append(dom_freq)
            
            # Normalize PSD to range 0-100
            psd_min = np.min(psd)
            psd_max = np.max(psd)
            psd_norm = 100 * (psd - psd_min) / (psd_max - psd_min + 1e-10)  # avoid division by 0
            normalized_psds.append(psd_norm)

    # BAND POWERS SECTION--------------------------------------------------------
    all_band_powers = []
    for ch_data in eeg_data_filtered:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=1024, noverlap=512)
        valid_band = (freqs >= 1) & (freqs <= 50)
        freqs = freqs[valid_band]
        psd = psd[valid_band]
        
        if len(psd) > 0:
            band_powers = calculate_band_powers(freqs, psd)
            all_band_powers.append(band_powers)

    # Average across channels
    avg_band_powers = {
        'delta': np.mean([bp['delta'] for bp in all_band_powers]),
        'theta': np.mean([bp['theta'] for bp in all_band_powers]),
        'alpha': np.mean([bp['alpha'] for bp in all_band_powers]),
        'beta': np.mean([bp['beta'] for bp in all_band_powers]),
        'gamma': np.mean([bp['gamma'] for bp in all_band_powers]),
    }

    # Update temporal buffer
    buffer.append(avg_band_powers)
    
    # Calculate smoothed values
    smoothed_powers = {
        band: np.mean([b[band] for b in buffer])
        for band in avg_band_powers.keys()
    }
 
    dominant_band = max(smoothed_powers, key=smoothed_powers.get)

    # Calculate average metrics
    if dominant_freqs:
        average_dominant_freq = np.mean(dominant_freqs)
        
        avg_normalized_psd = np.mean([np.mean(psd) for psd in normalized_psds]) if normalized_psds else 0

        # Find wave type
        wave_type = classify_wave(average_dominant_freq)

        # Prepare data for sending to Render server
        eeg_data_to_send = {
            "wave_type": wave_type,
            "dominant_freq": float(average_dominant_freq),
            'dominant_band': dominant_band,
            'intensity': smoothed_powers[dominant_band],
            "psd": float(avg_normalized_psd),
            "timestamp": time.time()
        }
        
        # Send the processed data to the Render server
        send_data_to_render(eeg_data_to_send)
        
        return eeg_data_to_send
    
    return None

def main():
    # Load the full dataset
    eeg_eog_data = pd.read_csv(r"C:\Users\carol\Documents\VSPrograms\Signal_Processing\Rehearsal_031322\Subject1\EEG\D1_EEG_EOG.csv", header=None)

    # Initialize temporal buffer
    smoothing_buffer = deque(maxlen=BUFFER_SIZE)
    
    # Split EEG and EOG data
    eeg_data = eeg_eog_data.iloc[:-4, :].values  # 28 EEG channels
    eog_data = eeg_eog_data.iloc[-4:, :].values  # 4 EOG channels
    
    # Define channel names
    eeg_ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','C3','Cz','C4','CP5','CP1','CP2','CP6',
               'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
    
    # Calculate the total number of samples
    total_samples = eeg_data.shape[1]
    
    # Process data in chunks of 1000 samples
    for start_idx in range(0, total_samples, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, total_samples)
        
        print(f"Processing samples {start_idx} to {end_idx} of {total_samples}")
        
        # Extract current chunk
        eeg_chunk = eeg_data[:, start_idx:end_idx]
        eog_chunk = eog_data[:, start_idx:end_idx]
        
        # Process this chunk
        result = process_eeg_chunk(eeg_chunk, eog_chunk, eeg_ch_names, smoothing_buffer)

        # Optional: Add a small delay to prevent overwhelming the app
        #time.sleep(0.1)  # Adjust as needed
        
        # You could also add logic here to wait for confirmation from the app
        # before processing the next chunk

if __name__ == "__main__":
    main()