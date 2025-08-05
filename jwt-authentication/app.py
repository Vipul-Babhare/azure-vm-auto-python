from flask import Flask, jsonify
import jwt
import datetime
import os

app = Flask(__name__)
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

@app.route('/token', methods=['POST'])
def generate_token():
    # No need to read any input
    payload = {
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return jsonify({"token": token})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
