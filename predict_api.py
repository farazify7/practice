from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

app = Flask(__name__)

# Train model on iris data
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # Expect a list of features
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Example dummy response
    return jsonify({"prediction": "setosa"})

# ✅ Add this block at the bottom
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var or default to 5000
    app.run(host="0.0.0.0", port=port)
