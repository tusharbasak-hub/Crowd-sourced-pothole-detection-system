import time
import pandas as pd
from ml_sender import send_prediction
from collections import Counter
import random
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# Your SEQUENCE_LENGTH (Window length)
SEQUENCE_LENGTH = 10 
# Data file containing sensor readings and GPS
DATA_FILE = '..\data\combined.csv' 
# Frequency of sending data (seconds)
SEND_INTERVAL_SECONDS = 1
# Filename for your saved PyTorch model
MODEL_FILE = '..\src\best_roadcnn.pth' 
# Features used for the model (Accelerometer and Gyroscope)
FEATURES = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
BACKEND_URL = "http://10.149.131.154:5000/update"


# Global variables for the model and scaler
model = None
scaler = None 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# IMPORTANT: You must ensure this class definition exactly matches the CNN architecture
# you used during your training (including all Conv1D, Pool, and Linear layer sizes).
class RoadConditionCNN(nn.Module):
    def __init__(self, input_size=len(FEATURES), num_classes=2):
        super(RoadConditionCNN, self).__init__()
        # Input shape expected: (Batch, Features=6, Sequence=10)
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Output sequence length: 5
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Output sequence length: 2 (since floor(5/2) = 2)
        
        # Flattened size: channels * final_sequence_length (128 * 2 = 256)
        flattened_size = 128 * 2 
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        x = x.view(x.size(0), -1) # Flatten (Batch_size, 256)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc_out(x)
        return x

def load_model_and_scaler(model_path: str, data_for_scaler: pd.DataFrame):
    """
    Initializes and loads the PyTorch model and fits the StandardScaler
    on the entire dataset (which is the required behavior for deployment, 
    even if the ideal method is loading a pre-fitted scaler object).
    """
    global model, scaler
    
    # 1. Initialize and Fit the Scaler (CRITICAL for ML inference)
    # The scaler must be fitted on the training data's statistics. Here we fit
    # on the full dataset to ensure prediction is possible.
    print("Initializing StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(data_for_scaler[FEATURES].values)
    
    # 2. Load the Model
    try:
        model = RoadConditionCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode for inference
        print(f"✅ Successfully loaded PyTorch model from {model_path} and set to {device}.")
    except FileNotFoundError:
        print(f"🚨 Error: Model file '{model_path}' not found.")
        print("    -> Falling back to **Simulation Mode** (using ground truth labels from CSV).")
        model = None
    except Exception as e:
        print(f"🚨 Error loading model: {e}")
        print("    -> Falling back to **Simulation Mode** (using ground truth labels from CSV).")
        model = None


def run_prediction_cycle(df: pd.DataFrame):
    """
    Main loop that handles windowing, prediction, and sending data.
    """
    current_idx = 0
    window_count = 0

    print(f"\n--- Starting Real-Time Prediction and Sender Loop ---")
    print(f"Sending data every {SEND_INTERVAL_SECONDS} seconds to {BACKEND_URL}")

    while True:
        # 1. Define the current window slice
        window_end_idx = current_idx + SEQUENCE_LENGTH
        
        # Check if we have enough data for a full window
        if window_end_idx > len(df):
            print("\nEnd of simulated data reached. Restarting simulation from the beginning.")
            current_idx = 0 # Loop back to the start
            continue

        # Get the current window of sensor data
        df_window = df.iloc[current_idx:window_end_idx]

        # 2. Extract GPS coordinates (from the last reading in the window)
        last_reading = df_window.iloc[-1]
        lat = last_reading['latitude']
        lon = last_reading['longitude']

        # 3. Perform Prediction (Actual or Simulated)
        prediction_label = None
        true_label = Counter(df_window['roadCondition']).most_common(1)[0][0] # Used for logging/fallback
        
        if model is not None and scaler is not None:
            # --- ACTUAL ML PREDICTION LOGIC ---
            try:
                # 3a. Extract features and standardize
                X_window = df_window[FEATURES].values
                X_scaled = scaler.transform(X_window)
                
                # 3b. Reshape for CNN: (1, N_FEATURES, SEQUENCE_LENGTH)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).T.unsqueeze(0).to(device)
                
                # 3c. Predict
                with torch.no_grad():
                    output = model(X_tensor)
                    _, predicted_class = torch.max(output, 1)
                    
                # 3d. Post-process (0=Good, 1=Bad or vice versa based on your LabelEncoder)
                # Assuming 0 maps to 'Good' and 1 maps to 'Bad'
                prediction_label = "Bad" if predicted_class.item() == 1 else "Good"
                
            except Exception as e:
                print(f"🚨 Prediction failed during inference: {e}. Using fallback label.")
        
        # Fallback to simulation if model failed or was not loaded
        if prediction_label is None:
            prediction_label = "Good" if true_label == "Good" else "Bad"


        window_count += 1
        
        print(f"\n[Window #{window_count}]")
        print(f"  > Model Status: {'Active' if model else 'SIMULATION'}")
        print(f"  > Final Prediction Sent: {prediction_label}")
        print(f"  > True Condition (from CSV): {true_label}")
        print(f"  > Location: {lat}, {lon}")

        # 4. Send the JSON payload to the backend
        send_prediction(lat, lon, prediction_label)

        # 5. Advance the index and wait
        current_idx += SEQUENCE_LENGTH
        time.sleep(SEND_INTERVAL_SECONDS)

# --- Main Execution Function ---
def main():
    """
    Orchestrates the loading, initialization, and prediction loop.
    """
    if not os.path.exists(DATA_FILE):
        print(f"🚨 Error: Data file '{DATA_FILE}' not found. Please ensure it is in the same directory.")
        return

    try:
        # Load the entire dataset
        df = pd.read_csv(DATA_FILE)
        print(f"Successfully loaded {len(df)} rows from {DATA_FILE}")

        required_cols = FEATURES + ['latitude', 'longitude', 'roadCondition']
        if not all(col in df.columns for col in required_cols):
            print(f"🚨 Error: Data is missing required columns. Needs: {required_cols}")
            return

        # 0. Load the Model and Scaler (Initial Setup)
        load_model_and_scaler(MODEL_FILE, df)

        # 1. Start the main prediction loop
        run_prediction_cycle(df)

    except Exception as e:
        print(f"\nFATAL ERROR in main execution: {e}")

if __name__ == '__main__':
    # Set the timezone environment variable for proper timestamping (optional, but good practice)
    os.environ['TZ'] = 'Asia/Kolkata' 
    # time.tzset()
    main()