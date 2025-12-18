# backend/server.py

from flask import Flask, jsonify
from threading import Thread
from inference import start_prediction_loop

app = Flask(__name__)
latest_data = {}


def update_latest(data):
    global latest_data
    latest_data = data
    print("Updated:", latest_data)


@app.route("/get_latest", methods=["GET"])
def get_latest():
    if latest_data:
        return jsonify(latest_data)
    return jsonify({"error": "no data yet"}), 404


if __name__ == "__main__":
    t = Thread(target=start_prediction_loop, args=(update_latest,), daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000)
