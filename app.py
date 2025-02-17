import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model, label_encoder, scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        math_score = float(request.form["math"])
        reading_score = float(request.form["reading"])
        writing_score = float(request.form["writing"])

        input_data = np.array([[math_score, reading_score, writing_score]])
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        return render_template("index.html", prediction=f"Predicted Race/Ethnicity: {predicted_class}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
