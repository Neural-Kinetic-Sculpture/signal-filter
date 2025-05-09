from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import os
import json
import threading
import random

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
    "alpha_band": -1,
    "beta_band": -1,
    "theta_band": -1,
    "delta_band": -1,
    "gamma_band": -1,
    "dominant_band": "none",
    "alpha_beta_ratio": -1,
    "alpha_delta_ratio": -1,
    "peak_alpha_freq": -1,
    "psd": -1,
    "timestamp": time.time(),
}

def generate_random_band_powers():
    """Generate random band powers that sum to exactly 100"""
    # Generate 5 random numbers
    bands = [random.uniform(5, 40) for _ in range(5)]
    
    # Normalize to sum to exactly 100
    total = sum(bands)
    normalized_bands = [round((b / total) * 100, 2) for b in bands]
    
    # Ensure they sum to exactly 100 (may be slightly off due to rounding)
    adjustment = 100 - sum(normalized_bands)
    normalized_bands[0] += adjustment
    
    # Determine dominant band index (highest value)
    dominant_idx = normalized_bands.index(max(normalized_bands))
    band_names = ["alpha", "beta", "theta", "delta", "gamma"]
    dominant_band = band_names[dominant_idx]
    
    # Calculate ratios
    alpha_band = normalized_bands[0]
    beta_band = normalized_bands[1]
    delta_band = normalized_bands[3]
    
    alpha_beta_ratio = round(alpha_band / beta_band, 2) if beta_band > 0 else 0
    alpha_delta_ratio = round(alpha_band / delta_band, 2) if delta_band > 0 else 0
    
    # Random peak alpha frequency between 8-12 Hz
    peak_alpha_freq = round(random.uniform(8, 12), 2)
    
    # Random PSD value
    psd = round(random.uniform(10, 200), 2)
    
    return {
        "alpha_band": normalized_bands[0],
        "beta_band": normalized_bands[1],
        "theta_band": normalized_bands[2],
        "delta_band": normalized_bands[3],
        "gamma_band": normalized_bands[4],
        "dominant_band": dominant_band,
        "alpha_beta_ratio": alpha_beta_ratio,
        "alpha_delta_ratio": alpha_delta_ratio,
        "peak_alpha_freq": peak_alpha_freq,
        "psd": psd,
        "timestamp": time.time(),
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
            
            # Create randomized default data with current timestamp
            default_data = generate_random_band_powers()
            
            # Only broadcast if we have clients connected
            if clients_connected > 0:
                socketio.emit('eeg_data', json.dumps(default_data))
                print(f"Broadcasting randomized default data: {default_data['alpha_band']:.1f}α, {default_data['beta_band']:.1f}β, {default_data['theta_band']:.1f}θ, {default_data['delta_band']:.1f}δ, {default_data['gamma_band']:.1f}γ")
        
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
                "alpha_band": data.get("alpha_band", 0),
                "beta_band": data.get("beta_band", 0),
                "theta_band": data.get("theta_band", 0),
                "delta_band": data.get("delta_band", 0),
                "gamma_band": data.get("gamma_band", 0),
                "dominant_band": data.get("dominant_band", 0),
                "alpha_beta_ratio": data.get("alpha_beta_ratio", 0),
                "alpha_delta_ratio": data.get("alpha_delta_ratio", 0),
                "peak_alpha_freq": data.get("peak_alpha_freq", 0),
                "psd": data.get("psd", 0),
                "timestamp": last_real_data_time,
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
        # Send randomized default data
        default_data = generate_random_band_powers()
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

@app.route('/test_command', methods=['GET'])
def test_command():
    """Send a test control command to all clients"""
    test_array = ["TEST_FROM_SERVER", "0", "0", "5", "1", "100", "FF0000"]
    socketio.emit('control_command', test_array)
    return jsonify({"status": "command sent"})

@socketio.on('control_command')
def handle_control_command(data):
    print(f"➡️ Received control command: {data}")
    print(f"Type: {type(data)}")
    print(f"Connected clients: {clients_connected}")
    
    # Log parsed information if it's an array
    if isinstance(data, list):
        try:
            if len(data) >= 6:
                row, col, speed, direction, brightness, color = data[:6]
                print(f"✨ Parsed array command - Panel: [{row},{col}], Speed: {speed}, Direction: {'up' if direction == '1' else 'down'}, Brightness: {brightness}, Color: {color}")
            else:
                print(f"⚠️ Array command format incorrect, expected at least 6 parameters")
        except Exception as e:
            print(f"⚠️ Error parsing array command: {e}")
    # Backward compatibility for string commands
    elif isinstance(data, str):
        try:
            parts = data.split()
            if len(parts) >= 6:
                row, col, speed, direction, brightness, color = parts[:6]
                print(f"✨ Parsed string command - Panel: [{row},{col}], Speed: {speed}, Direction: {'up' if direction == '1' else 'down'}, Brightness: {brightness}, Color: {color}")
            else:
                print(f"⚠️ Command format incorrect, expected at least 6 parameters")
        except Exception as e:
            print(f"⚠️ Error parsing command: {e}")
    
    # Re-emit the command exactly as received (no transformation)
    socketio.emit('control_command', data)
    
    # Return acknowledgment
    return "command received"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)