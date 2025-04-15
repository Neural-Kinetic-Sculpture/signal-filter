# EEG Signal Processing Pipeline

## Overview
This repository contains a signal processing pipeline for EEG data analysis using Python and MNE-Python. The script loads EEG data, filters it, and visualizes raw signals and power spectral density (PSD).

## Features
- Loads EEG data from a CSV file
- Extracts EEG channels from combined EEG-EOG data
- Creates an MNE Raw object for further analysis
- Applies high-pass and low-pass filtering
- Visualizes raw EEG signals
- Computes and plots power spectral density (PSD)

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas matplotlib mne
```

## Usage
1. Clone the repository:
2. Place your EEG data CSV file in the appropriate directory.
3. Modify the file path in the script to point to your EEG dataset.
4. Run the script:
   ```bash
   python pipeline.py
   ```

## File Structure
```
.
├── pipeline.py  # Main EEG processing script
├── signal_to_app.py  # Websocket connection to app
└── README.md             # Documentation
```

## Data Format
The script expects a CSV file containing EEG data, where:
- The first 28 rows represent EEG channels.
- The last 4 rows represent EOG channels.
- Sampling frequency is set to **1000 Hz**.

## Filtering
- **High-pass filter**: Removes slow drifts below **1 Hz**.
- **Low-pass filter**: Removes high-frequency noise above **50 Hz**.

## Visualization
The script provides two types of plots:
1. **Raw EEG signal** visualization over time.
2. **Power Spectral Density (PSD)** plot to analyze frequency components.

## Libraries needed:
pip install flask flask-socketio numpy pandas mne eventlet flask-cors scipy
pip install pylsl
