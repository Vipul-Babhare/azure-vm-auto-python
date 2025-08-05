from flask import Flask, request, jsonify
import jwt
import os
import requests

app = Flask(__name__)
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://localhost:8501/v1/models/your_model:predict")

@app.route('/predict', methods=['POST'])
def proxy_predict():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or malformed token"}), 401

    token = auth_header.split(" ")[1]

    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

    # Forward the prediction request
    response = requests.post(TF_SERVING_URL, json=request.get_json())
    return jsonify(response.json()), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # ✅ FIXED to 5000
