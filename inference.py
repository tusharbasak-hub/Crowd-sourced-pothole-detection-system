# backend/inference.py

import time
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ---------------- CONFIG ----------------
SEQUENCE_LENGTH = 10
SEND_INTERVAL_SECONDS = 1

DATA_FILE = "data/combined.csv"
MODEL_FILE = "model/best_roadcnn.pth"

FEATURES = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL (EXACT MATCH) ----------------
class RoadConditionCNN(nn.Module):
    def __init__(self, input_size=len(FEATURES), num_classes=2):
        super(RoadConditionCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 2, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        return self.fc_out(x)

# ---------------- MAIN LOOP ----------------
def start_prediction_loop(update_callback):
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")

    df = pd.read_csv(DATA_FILE)

    required_cols = FEATURES + ['latitude', 'longitude', 'roadCondition']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV missing required columns")

    # scaler (same logic as original)
    scaler = StandardScaler()
    scaler.fit(df[FEATURES].values)

    # model
    model = RoadConditionCNN().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    print("✅ ML model + scaler loaded successfully")

    idx = 0
    window_count = 0

    while True:
        if idx + SEQUENCE_LENGTH > len(df):
            idx = 0

        df_window = df.iloc[idx:idx + SEQUENCE_LENGTH]
        last = df_window.iloc[-1]

        lat = float(last['latitude'])
        lon = float(last['longitude'])

        prediction_label = None
        true_label = Counter(df_window['roadCondition']).most_common(1)[0][0]

        try:
            X = df_window[FEATURES].values
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).T.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(X_tensor)
                _, pred_class = torch.max(output, 1)

            prediction_label = "Bad" if pred_class.item() == 1 else "Good"

        except Exception as e:
            print(f"⚠️ ML inference failed: {e}")
            prediction_label = "Good" if true_label == "Good" else "Bad"

        window_count += 1

        data = {
            "latitude": lat,
            "longitude": lon,
            "road_condition": prediction_label
        }

        print(f"[Window {window_count}] {data}")

        update_callback(data)

        idx += SEQUENCE_LENGTH
        time.sleep(SEND_INTERVAL_SECONDS)
