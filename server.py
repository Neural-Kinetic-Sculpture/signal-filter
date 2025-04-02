from flask import Flask, request
from flask_socketio import SocketIO
import numpy as np
import pandas as pd
import mne
import eventlet

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def home():
    return "EEG Flask WebSocket Server is Running!"

# Function to simulate EEG data streaming
def stream_eeg_data():
    while True:
        # Simulated PSD & Frequency Data
        psd_value = np.random.uniform(1.0, 5.0)  # Random PSD value
        freq_value = np.random.choice([8, 10, 12])  # Simulated Alpha frequency

        data = {"psd": psd_value, "frequency": freq_value}

        # Send Data to React Native App via WebSockets
        socketio.emit('eeg_data', data)
        
        eventlet.sleep(1)  # Send data every second

# Start streaming when WebSocket client connects
@socketio.on('connect')
def handle_connect():
    print("Client connected!")
    eventlet.spawn(stream_eeg_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=10000)
