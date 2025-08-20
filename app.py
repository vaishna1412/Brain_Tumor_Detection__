from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import cv2
import os

app = Flask(__name__)
cnn_model = load_model('models/cnn_model.h5')
ml_models = joblib.load('models/ml_models.pkl')
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')  # just shows the form

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    model_choice = request.form['model']
    path = os.path.join('uploads', file.filename)
    file.save(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (100, 100))

    mean_intensity = np.mean(img_resized)
    variance = np.var(img_resized)

    if mean_intensity < 20 or variance < 100:
        os.remove(path)
        return jsonify({'prediction': '⚠️ Please upload a valid brain MRI image.'})

    edges = cv2.Canny(img_resized, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if model_choice == 'cnn':
        input_img = img_resized.reshape(1, 100, 100, 1) / 255.0
        pred = cnn_model.predict(input_img)
        result = 'Tumor' if np.argmax(pred) == 0 else 'No Tumor'
    else:
        flat = img_resized.flatten().reshape(1, -1) / 255.0
        pca = ml_models['pca']
        reduced = pca.transform(flat)
        model = ml_models[model_choice]
        prediction = model.predict(reduced)
        result = 'Tumor' if prediction[0] == 0 else 'No Tumor'

    os.remove(path)
    return jsonify({'prediction': result})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
