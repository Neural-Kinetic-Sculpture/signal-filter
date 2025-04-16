from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import os
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", json=json)
CORS(app)
API_KEY = os.environ['EEG_API_KEY']

clients_connected = 0
thread = None

# Store the latest EEG data
latest_eeg_data = {
    "dominant_freq": 0,
    "psd": 0,
    "timestamp": time.time(),
    "confidence": 0.8  # Default value
}

@app.route('/')
def index():
    return "🎛️ Neural Kinetic Sculpture EEG WebSocket Server Running"

@app.route('/receive_eeg_data', methods=['POST'])
def receive_eeg_data():
    global latest_eeg_data
    
    # Verify API key
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    if request.method == 'POST':
        try:
            data = request.json
            
            # Update our latest data
            latest_eeg_data = {
                "dominant_freq": data.get("dominant_freq", 0),
                "psd": data.get("psd", 0),
                "timestamp": data.get("timestamp", time.time()),
                "confidence": 0.85,  # Placeholder for now
                "wave_type": data.get("wave_type", 0)
            }
            
            # Broadcast to all connected clients
            socketio.emit('eeg_data', json.dumps(latest_eeg_data))
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

@socketio.on('connect')
def handle_connect():
    global thread, clients_connected
    clients_connected += 1
    print(f'Client connected: {clients_connected}')
    socketio.emit('test_response', '✅ Connected to EEG WebSocket server')
    
    # Send the latest data to newly connected client
    socketio.emit('eeg_data', json.dumps(latest_eeg_data))

@socketio.on('disconnect')
def handle_disconnect():
    global clients_connected
    clients_connected = max(0, clients_connected - 1)
    print(f'Client disconnected. Remaining: {clients_connected}')

@socketio.on('test_connection')
def handle_test(message):
    print(f"Test connection message received: {message}")
    socketio.emit('test_response', '✅ Echo from server')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)