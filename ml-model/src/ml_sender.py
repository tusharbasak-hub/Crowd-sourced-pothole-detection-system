import requests
import json
import time

# IMPORTANT: Replace this IP with your actual backend IP if it changes.
BACKEND_URL = "http://10.149.131.154:5000/update"

def send_prediction(lat: float, lon: float, prediction: str):
    """
    Constructs the JSON payload and sends a POST request to the backend server.
    
    Args:
        lat (float): The latitude of the road condition measurement.
        lon (float): The longitude of the road condition measurement.
        prediction (str): The predicted road condition ("Good" or "Bad").
    """
    payload = {
        "latitude": lat,
        "longitude": lon,
        "road_condition": prediction
    }
    
    try:
        # Send the POST request with the JSON payload
        response = requests.post(BACKEND_URL, json=payload, timeout=5)
        
        # Check for successful response status codes (2xx)
        if response.status_code == 200:
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Data successfully sent to backend.")
            # print(f"Response: {response.json()}") # Uncomment if the backend returns a JSON response
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Failed to send data. Status code: {response.status_code}")
            print(f"Backend response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Connection Error: Could not reach the backend server at {BACKEND_URL}.")
    except requests.exceptions.Timeout:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Timeout Error: Request to backend took too long.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Example usage when running ml_sender.py directly:
    print("Running a test prediction send (This will likely fail unless your backend is running at the specified IP).")
    test_lat, test_lon, test_pred = 28.7041, 77.1025, "Bad"
    send_prediction(test_lat, test_lon, test_pred)
    print("Test finished.")