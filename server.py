from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import os
import json
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", json=json)
CORS(app)
API_KEY = os.environ['EEG_API_KEY']

clients_connected = 0
broadcast_thread = None
last_real_data_time = 0
broadcasting_default = True

# Store the latest EEG data
latest_eeg_data = {
    "wave_type": "none",
    "dominant_freq": 0,
    "psd": 0,
    "timestamp": time.time(),
    "confidence": 0.0  # Default value
}

def background_broadcast():
    """Background thread that broadcasts default data when no real data is coming in"""
    global latest_eeg_data, broadcasting_default, last_real_data_time
    
    while True:
        current_time = time.time()
        
        # If we haven't received real data in 5 seconds, send default data
        if current_time - last_real_data_time > 5:
            if not broadcasting_default:
                print("Switching to default data broadcast mode")
                broadcasting_default = True
            
            # Create default data with current timestamp
            default_data = {
                "wave_type": "none",
                "dominant_freq": 0,
                "psd": 0,
                "timestamp": current_time,
                "confidence": 0.0
            }
            
            # Only broadcast if we have clients connected
            if clients_connected > 0:
                socketio.emit('eeg_data', json.dumps(default_data))
                print("Broadcasting default data")
        
        # Sleep to control broadcast rate (send every 1 second)
        time.sleep(1)

@app.route('/')
def index():
    return "🎛️ Neural Kinetic Sculpture EEG WebSocket Server Running"

@app.route('/receive_eeg_data', methods=['POST'])
def receive_eeg_data():
    global latest_eeg_data, last_real_data_time, broadcasting_default
    
    # Verify API key
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    if request.method == 'POST':
        try:
            data = request.json
            
            # Update timestamp for real data received
            last_real_data_time = time.time()
            
            if broadcasting_default:
                print("Received real data, switching to real data broadcast mode")
                broadcasting_default = False
            
            # Update our latest data
            latest_eeg_data = {
                "wave_type": data.get("wave_type", "unknown"),
                "dominant_freq": data.get("dominant_freq", 0),
                "dominant_band": data.get("dominant_band", "unknown"),
                "intensity": data.get("intensity", 0),
                "psd": data.get("psd", 0),
                "timestamp": last_real_data_time
            }
            
            # Broadcast to all connected clients
            socketio.emit('eeg_data', json.dumps(latest_eeg_data))
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

@socketio.on('connect')
def handle_connect():
    global broadcast_thread, clients_connected
    clients_connected += 1
    print(f'Client connected: {clients_connected}')
    socketio.emit('test_response', '✅ Connected to EEG WebSocket server')
    
    # Start background thread if not already running
    if broadcast_thread is None:
        broadcast_thread = threading.Thread(target=background_broadcast)
        broadcast_thread.daemon = True
        broadcast_thread.start()
        print("Started background broadcast thread")
    
    # Send the current data to newly connected client
    if broadcasting_default:
        # Send default data
        default_data = {
            "wave_type": "none",
            "dominant_freq": 0,
            "dominant_band": "none",
            "intensity": 0,
            "psd": 0,
            "timestamp": time.time(),
            "confidence": 0.0
        }
        socketio.emit('eeg_data', json.dumps(default_data))
    else:
        # Send latest real data
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

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check server status and data mode"""
    return jsonify({
        "status": "running",
        "clients_connected": clients_connected,
        "broadcasting_default": broadcasting_default,
        "last_real_data": time.time() - last_real_data_time
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)