# Crowd-Sourced Pothhole Detection System (NoHole)

This repository contains a **smartphone-based pothole detection system**.  
The Android app records motion sensor data while driving/riding, and a Python backend can process that data to classify road quality.

---

## 🚀 Project Overview

- Uses built-in mobile sensors (accelerometer, gyroscope)
- Detects potholes or unusual road vibration patterns
- Sends data to backend server through REST API
- Future support for ML inference

---

## 📱 Android App (Frontend)

**Folder:** `app/`

### Features
- Real-time sensor data capture
- Road anomaly detection logic
- Send data to backend
- MVVM architecture
- Retrofit / HTTP client (if integrated)

---

## 🖥 Backend Server (API)

**Folder:** `server/`

- Flask REST API
- Receives JSON sensor payload
- Returns prediction/dummy output

### ⚠️ ML Model Status
> I currently **do not have the ML model code or inference files**.  
> These are supervised by my instructor and will be added later when available.



## App Screenshots
<img width="350" height="700" alt="Screenshot_20251121_113801" src="https://github.com/user-attachments/assets/0f4fb857-d966-43a3-82ab-503cfff34d6c" />
<img width="350" height="700" alt="Screenshot_20251121_114250" src="https://github.com/user-attachments/assets/e0778536-78e4-41b4-90ad-e699d472cd11" />
<img width="2160" height="1120" alt="Working_interface" src="https://github.com/user-attachments/assets/e065a089-63af-42f0-8f1e-f6dd34bc6294" />




