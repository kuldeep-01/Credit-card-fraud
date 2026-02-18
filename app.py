from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained pipeline
model = joblib.load("fraud_detection_pipeline.pkl")

# Feature names (must match training)
FEATURES = [
    'Time',  # Added Time
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route (form)
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form
        input_data = [float(request.form[feature]) for feature in FEATURES]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            result = f"⚠️ Fraud Detected (Probability: {probability:.2f})"
        else:
            result = f"✅ Legitimate Transaction (Probability: {probability:.2f})"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# API route (for JSON requests)
@app.route("/predict_api", methods=["POST"])
def predict_api():

    data = request.get_json(force=True)

    input_df = pd.DataFrame([data], columns=FEATURES)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    })

# Run app
if __name__ == "__main__":
    app.run(debug=True)
