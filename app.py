from flask import Flask, render_template, request, url_for, Response
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import convolute_lib as cnn
import pytesseract
import os

from flask_cors import CORS

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract'
app = Flask(__name__, static_folder='assets')
CORS(app)
dic = {0: 'dog', 1: 'cat'}

model = load_model('my_model.h5')
def predict_label(img_path):
        i = image.load_img(img_path, target_size=(224, 224))
        i = image.img_to_array(i)
        i = i.reshape(1, 224, 224, 3)
        p = model.predict(i)
        return (dic[int(p[0][0])])

# model.predict(cv2.imread("train/test"));
@app.route('/')
def index():
    return "<h1>Tool 4 Image</h1>"
@app.route('/sharp', methods=['POST'])
def sharp():
    imageFile = request.files.get('file', '')
    imageFile.save(os.path.join("uploads", imageFile.filename))

    img = cv2.imread(os.path.join("uploads", imageFile.filename))

    kernel = np.array(([0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]))
    im = cv2.filter2D(img, -1, kernel)
    filename = 'assets/savedImage.jpg'
    cv2.imwrite(filename, im)
    path = request.url_root+"assets/savedImage.jpg";
    print(path)
    return path;

    # construct average blurring kernels used to smooth an image
    smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

    # construct a sharpening filter
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]))

    filters = [
        ("Sharpen", sharpen),
    ]

    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    for i, filter in enumerate(filters):
        axes = fig.add_subplot(1, 3, i + 1)
        axes.set(title=filter[0])
        axes.grid(False)
        axes.set_xticks([])
        axes.set_yticks([])
        # img_out = cnn.convolve_np4(img, filter[1])
        img_out = cv2.filter2D(img, -1, filter[1])
        backtorgb = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)

        axes.imshow(backtorgb, cmap=None, vmin=0, vmax=255)
    plt.show()
    return "<h1>sharp 4 Image</h1>"

@app.route('/classification', methods=['POST'])
def classification():
    imageFile = request.files.get('file', '')
    imageFile.save(os.path.join("uploads", imageFile.filename))
    # img_path = 'meo1.jpg';

    p = predict_label(os.path.join("uploads", imageFile.filename));
    return str(p)

@app.route('/upload', methods=['POST'])
def upload():
    imageFile = request.files.get('file','')
    image = Image.open(imageFile)
    text = pytesseract.image_to_string(image)
    f = open("sample.txt", "a")
    f.truncate(0)
    f.write(text)
    f.close()
    return text

if __name__ == "__main__":
    app.run(debug=True)
