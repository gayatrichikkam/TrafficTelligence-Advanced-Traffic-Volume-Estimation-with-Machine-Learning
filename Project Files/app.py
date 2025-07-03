from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        hours = int(request.form['hours'])
        month = int(request.form['month'])

        # Create feature vector (must match training order)
        input_features = np.array([[temp, rain, snow, hours, month]])
        prediction = model.predict(input_features)[0]

        return render_template('result.html', result=f"Estimated Traffic Volume: {round(prediction)}")
    except Exception as e:
        return render_template('result.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
