from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import random
import threading
import os
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", json=json)
CORS(app)

clients_connected = 0
thread = None

def simulate_realistic_alpha_wave():
    # Simulate frequency drift in alpha range
    dominant_freq = round(random.gauss(10.3, 0.6), 2)  # ~10.3 Hz ± 0.6
    dominant_freq = max(8.0, min(dominant_freq, 12.0))  # Clamp to alpha band

    # Simulate PSD (signal power), e.g. 0-100
    psd = round(random.uniform(0.0, 100.0), 2)

    # Confidence is ratio of alpha power to total brain power
    confidence = round(random.uniform(0.7, 0.95), 2)

    eeg_data = {
        "wave_type": "alpha",
        "dominant_freq": dominant_freq,
        "psd": psd,
        "confidence": confidence,
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
            eeg_data = simulate_realistic_alpha_wave()
            print("📡 Emitting EEG Data:", eeg_data)
            socketio.emit('eeg_data', json.dumps(eeg_data))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
