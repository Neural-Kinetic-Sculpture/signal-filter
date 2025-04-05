from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import random
import math
import threading
import os
import json
thread = None

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", json=json)
CORS(app)

clients_connected = 0  # track number of connected clients

def generate_alpha_wave(frequency=10, amplitude=50):
    t = time.time()
    frequency = round(random.uniform(8, 13), 2)
    raw_amplitude = amplitude * math.sin(2 * math.pi * frequency * t)
    noise = random.uniform(-10, 10)
    value = round(raw_amplitude + noise, 2)
    psd = round(abs(value) + random.uniform(0, 5), 2)

    eeg_data = {
        "wave_type": "alpha",
        "frequency": frequency,
        "amplitude": value,
        "psd": psd,
        "timestamp": round(t, 3)
    }

    try:
        json.dumps(eeg_data)  # Validate it's JSON serializable
    except TypeError as e:
        print("Serialization error:", e)
        return None

    return eeg_data

@app.route('/')
def index():
    return "EEG WebSocket Server Running"

@socketio.on('connect')
def handle_connect():
    global thread, clients_connected
    clients_connected += 1
    print('Client connected:', clients_connected)
    
    # Send a test message to verify connection
    socketio.emit('test_response', 'Hello from Flask server')
    
    # Send a test EEG data immediately to verify data format works
    test_data = {
        "wave_type": "test",
        "frequency": 10.0,
        "amplitude": 15.0,
        "psd": 20.0,
        "timestamp": time.time()
    }
    socketio.emit('eeg_data', json.dumps(test_data))
    
    # Start background thread only if it isn't already running
    if thread is None:
        thread = socketio.start_background_task(send_eeg_data)

@socketio.on('disconnect')
def handle_disconnect():
    global clients_connected
    clients_connected = max(0, clients_connected - 1)
    print('Client disconnected. Remaining:', clients_connected)

@socketio.on('test_connection')
def handle_test(message):
    print(f"Test connection message received: {message}")
    # Echo back to verify two-way communication
    socketio.emit('test_response', 'Hello from Flask server')

def send_eeg_data():
    """Background task that sends EEG data to clients"""
    while True:
        socketio.sleep(1)  # Use socketio.sleep instead of time.sleep
        if clients_connected > 0:
            eeg_data = generate_alpha_wave()
            if eeg_data:
                print("✅ Emitting EEG Data:", eeg_data)
                # Emit from this thread in a thread-safe way
                socketio.emit('eeg_data', json.dumps(eeg_data))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
