from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Dictionary for class labels
dic = {0: 'Normal', 1: 'Doubtful', 2: 'Mild', 3: 'Moderate', 4: 'Severe'}

# Image Size
img_size = 256
model = load_model(r"D:\0test\Flask-Knee-Osteoarthritis-Classification\model1.h5")

model.make_predict_function()

def predict_label(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized = cv2.resize(gray, (img_size, img_size)) 
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)  # Ensure this matches the model's expected input shape
    p = model.predict(i)
    class_index = np.argmax(p, axis=1)[0]
    return dic[class_index]


# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe Artificial Intelligence Hub..!!!"

@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        img = request.files['file']
        if img.filename == '':
            return "No selected file", 400
        
        filename = secure_filename(img.filename)
        img_path = os.path.join('uploads', filename)
        img.save(img_path)
        
        try:
            p = predict_label(img_path)
            # Optionally remove the file after prediction
            os.remove(img_path)
            return str(p).lower()
        except Exception as e:
            return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
