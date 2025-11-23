from flask import Flask, request, jsonify

app = Flask(__name__)

latest_data = {}

@app.route('/update', methods=['POST'])
def update():
    global latest_data
    latest_data = request.get_json()
    print("Received:", latest_data)
    return jsonify({"status": "received"}), 200

@app.route('/get_latest', methods=['GET'])
def get_latest():
    if latest_data:
        return jsonify(latest_data)
    else:
        return jsonify({"error": "no data yet"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
