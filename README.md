# Neural Kinetic Sculpture EEG Processing Pipeline

## Overview
This repository contains a signal processing pipeline for EEG data analysis using Python. The system captures EEG data from a LiveAmp device, processes it in real-time, and sends the extracted brain wave information to a web application via WebSockets.

## Authors
- Carolina Mims
- Katrina Viray

## Features
- Real-time EEG data acquisition from LiveAmp device via Lab Streaming Layer (LSL)
- Advanced signal processing with H-infinity filtering for EOG artifact removal
- High and low pass filtering to isolate relevant frequency bands
- Robust referencing using average reference
- Real-time brain wave classification (alpha, beta, delta, theta, gamma)
- WebSocket server for transmitting processed data to client applications
- Automatic reconnection handling for device disconnections
- Fallback to default data when no EEG stream is available

## System Architecture
1. **EEG Data Acquisition**: Captures raw EEG data from LiveAmp device
2. **Signal Processing**: Filters and processes the raw EEG data
3. **Feature Extraction**: Calculates dominant frequency and power spectral density
4. **WebSocket Server**: Broadcasts processed data to connected clients

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install flask flask-socketio numpy pandas mne eventlet flask-cors scipy pylsl
pip install scikit-learn
```

## Server Setup
The WebSocket server runs on Render and requires an API key for authentication.
1. Set environment variable: `EEG_API_KEY`
2. The server will run on port 5001 by default

## Signal Processing Pipeline
The pipeline performs several steps:
1. **Buffer Creation**: Accumulates EEG samples into a buffer of size 500
2. **H-infinity Filtering**: Removes EOG artifacts using reference signals
3. **High-pass Filter**: Removes slow drifts below 1 Hz
4. **Low-pass Filter**: Removes high-frequency noise above 50 Hz
5. **Average Reference**: Improves signal quality by referencing to the average
6. **Frequency Analysis**: Calculates power spectral density and dominant frequency
7. **Wave Classification**: Identifies brain wave types (delta, theta, alpha, beta, gamma)

## Brain Wave Classification
The system classifies brain waves into the following categories:
- **Delta**: 0.5-4 Hz (deep sleep)
- **Theta**: 4-8 Hz (drowsiness, meditation)
- **Alpha**: 8-12 Hz (relaxed awareness)
- **Beta**: 12-30 Hz (alert, focused)
- **Gamma**: 30-50 Hz (higher cognitive processing)

## Usage
1. Start the LiveAmp EEG stream
2. Run the signal processing script:
   ```bash
   python pipeline.py
   ```
3. Run the WebSocket server:
   ```bash
   python signal_to_app.py
   ```
4. Connect client applications to the WebSocket endpoint

## WebSocket API
The server provides the following endpoints:
- `/`: Status check endpoint
- `/receive_eeg_data`: POST endpoint to receive processed EEG data
- `/status`: GET endpoint to check server status and connection info

WebSocket events:
- `connect`: Fired when a client connects
- `disconnect`: Fired when a client disconnects
- `test_connection`: Test event to verify connection
- `eeg_data`: Event containing the processed EEG data

## Data Format
The WebSocket server sends JSON data with the following structure:
```json
{
  "wave_type": "alpha",
  "dominant_freq": 10.5,
  "psd": 75.3,
  "timestamp": 1681234567.89,
  "confidence": 0.85
}
```

## File Structure
```
.
├── pipeline.py         # EEG processing and LSL connection script
├── signal_to_app.py    # WebSocket server for data transmission
└── README.md           # Documentation
```

## EEG Channel Configuration
The system is configured for:
- 4 EEG channels (Fp1, Fp2, F3, F4)
- 4 EOG channels (EOG1, EOG2, EOG3, EOG4)
- Sampling rate: 1000 Hz

## Offline Analysis
The codebase also includes commented-out sections for offline analysis of pre-recorded EEG data. These can be useful for development and testing without a live EEG stream.

## Security
- API key authentication is required for posting data to the server
- CORS is enabled to allow cross-origin requests
- Environment variables are used for sensitive configuration

## Error Handling
- Automatic reconnection to LiveAmp if connection is lost
- Fallback to default data when no real data is available
- Timeout handling for connections to Render server
- Max client connection tracking

## Credits
This project uses a modified H-infinity filter implementation for EOG artifact removal.