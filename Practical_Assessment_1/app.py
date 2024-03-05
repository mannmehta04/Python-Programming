from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        p_cores = int(request.form['p_cores'])
        e_cores = int(request.form['e_cores'])
        base_speed = float(request.form['base_speed'])

        input_data = np.array([[p_cores, e_cores, base_speed]])
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
