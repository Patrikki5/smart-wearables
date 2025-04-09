import socket
import numpy as np
import joblib
from scipy.signal import find_peaks
import time

# Configuration
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8080
WINDOW_SIZE = 10  # Must match training
STEP_SIZE = 5     # 50% overlap
TIMEOUT = 10      # Seconds to wait for data

def safe_corr(x, y):
    """Safe correlation calculation handling constant inputs"""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.corrcoef(x, y)[0, 1]
        return 0 if np.isnan(r) else r

def extract_features(window_data):
    """Extract features matching training pipeline"""
    sensor_data = np.array([x[1:5] for x in window_data])  # Extract sensor values
    
    # Normalization (must match training)
    sensor_data = (sensor_data - 25) / (850 - 25)
    sensor_data = 1 - sensor_data  # Invert: 1 = pressure
    
    # Feature extraction
    mean = np.mean(sensor_data, axis=0)
    min_val = np.min(sensor_data, axis=0)
    
    valley_counts = [
        len(find_peaks(-sensor_data[:, i], height=-0.7)[0])
        for i in range(4)
    ]
    
    kick_detected = 1 if min_val[0] > 0.2 else 0
    correlation = safe_corr(sensor_data[:, 3], sensor_data[:, 2])
    diff_signal = sensor_data[:,3] - sensor_data[:,2]  # Heel - Toes
    # Frequency analysis (walking rhythm)
    fft = np.abs(np.fft.rfft(diff_signal))  # A3 (ball of foot)
    dominant_freq = np.argmax(fft[1:]) + 1 

    heel_valleys = len(find_peaks(-sensor_data[:,3], height=-0.5)[0])  # S4=Heel
    toe_peaks = len(find_peaks(sensor_data[:,2], height=0.3)[0])       # S3=Toes
    strike_ratio = heel_valleys / (toe_peaks + 1e-6)
    
    return np.concatenate(
                [mean, np.array([correlation]), min_val, valley_counts, 
                np.array([kick_detected, dominant_freq]),
                np.array([
                heel_valleys,             # Number of heel strikes
                toe_peaks,                # Number of toe-offs
                strike_ratio,               # Should be ~1 (1 heel strike per toe-off)
                np.mean(sensor_data[:,3] - sensor_data[:,2]),  # Heel-toe pressure difference
                np.std(diff_signal)         # Variability in stride
                ])])

def setup_server():
    """Configure and return socket server"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.settimeout(TIMEOUT)
    server_socket.listen(1)
    print(f"Server ready on {SERVER_IP}:{SERVER_PORT}")
    return server_socket

def process_data_stream(client_socket, model):
    buffer = []
    last_activity = None
    
    while True:
        try:
            # Read and clean data
            raw_data = client_socket.recv(1024).decode('utf-8').strip()
            if not raw_data:
                continue
                
            # Handle multiple lines/messages in one packet
            for line in raw_data.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Parse timestamp and sensor values
                    parts = line.split(',')
                    if len(parts) == 5:
                        # Clean each value (handle \r\n cases)
                        cleaned = [float(x.strip()) for x in parts]
                        buffer.append(cleaned)
                        
                        # Process when we have enough data
                        if len(buffer) >= WINDOW_SIZE:
                            window = buffer[-WINDOW_SIZE:]
                            features = extract_features(window)
                            
                            # Predict with confidence
                            prediction = model.predict(features.reshape(1, -1))[0]
                            proba = model.predict_proba(features.reshape(1, -1))[0]
                            confidence = np.max(proba)
                            
                            # Only update if confidence is high
                            if confidence > 0.5:  # Adjust threshold as needed
                                last_activity = prediction
                                print(f"Activity: {prediction} ({confidence:.2f}) | Sensors: {cleaned[1:]}")
                            
                            # Slide window
                            buffer = buffer[STEP_SIZE:] if len(buffer) > STEP_SIZE else []
                            
                except ValueError as e:
                    print(f"Skipping malformed data: {line} | Error: {e}")
                    continue
                    
        except socket.timeout:
            print("Connection timeout - waiting for data...")
            break
        except ConnectionResetError:
            print("Client disconnected")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

def main():
    # Load model
    try:
        model_data = joblib.load("gesture_classifier_with_metadata.pkl")
        model = model_data["model"]
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Set up server
    server_socket = setup_server()
    
    try:
        while True:
            print("Waiting for client connection...")
            try:
                client_socket, client_addr = server_socket.accept()
                client_socket.settimeout(TIMEOUT)
                print(f"Connected to {client_addr}")
                
                process_data_stream(client_socket, model)
                
            except socket.timeout:
                print("Server timeout - no connections")
                continue
            except Exception as e:
                print(f"Connection error: {e}")
                continue
            finally:
                if 'client_socket' in locals():
                    client_socket.close()
                print("Ready for new connection")
                
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server_socket.close()
        print("Server stopped")

if __name__ == "__main__":
    main()