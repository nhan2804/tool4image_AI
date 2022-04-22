from flask import Flask, render_template, request, url_for, Response
import cv2
import numpy as np
# from keras.models import load_model
# from keras.preprocessing import image
from PIL import Image
# import pytesseract
import os
import random
from flask_cors import CORS

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
app = Flask(__name__, static_folder='assets')
CORS(app)
dic = {1: 'dog', 0: 'cat'}


def kernel_generator(size):
    kernel = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            if i < j:
                kernel[i][j] = -1
            elif i > j:
                kernel[i][j] = 1
    return kernel

def tv_60(img,val,thresh):
    # cv2.namedWindow('image')
    # cv2.createTrackbar('val', 'image', 0, 255)
    # cv2.createTrackbar('threshold', 'image', 0, 100)
    # while True:
    thresh=int(thresh)
    val= int(val)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.getTrackbarPos('threshold', 'image')
    # val = cv2.getTrackbarPos('val', 'image')
    for i in range(height):
        for j in range(width):
            if np.random.randint(100) <= thresh:
                if np.random.randint(2) == 0:
                    gray[i, j] = min(gray[i, j] + np.random.randint(0, val+1), 255) # adding noise to image and setting values > 255 to 255.
                else:
                    gray[i, j] = max(gray[i, j] - np.random.randint(0, val+1), 0) # subtracting noise to image and setting values < 0 to 0.
    return gray;


# model = load_model('my_model.h5')
def predict_label(img_path):
        i = image.load_img(img_path, target_size=(224, 224))
        i = image.img_to_array(i)
        i = i.reshape(1, 224, 224, 3)
        # p = model.predict(i)
        # return (dic[int(p[0][0])])

# model.predict(cv2.imread("train/test"));
@app.route('/')
def index():
    return "<h1>Tool 4 Image</h1>"

def uploadFileImage(fileUpload,req):
    last_img = request.form.get('last_img',None)
    if last_img:
        print(last_img)
        image_file = cv2.imread(os.path.join("assets", last_img))
    else:
        fileUpload.save(os.path.join("uploads", fileUpload.filename))
        image_file = cv2.imread(os.path.join("uploads", fileUpload.filename))
    return image_file
def getPathFile(file,req):
    filename = 'assets/' + 'savedImage' + str(random.randint(0, 9999)) + '.jpg'
    cv2.imwrite(filename, file)
    path = req.url_root + filename
    return path


def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]).astype("uint8") # generating table for exponential function
    channel = cv2.LUT(channel, table)
    return channel
@app.route('/duo-tone', methods=['POST'])
def duo_tone():
    imageFile = request.files.get('file', '')
    exp = request.form['exponent']
    # switch = request.form['s1']
    img = uploadFileImage(imageFile,request)
    exp = int(exp)

    # cv2.createTrackbar('exponent', 'image', 0, 10, nothing)
    # switch1 = '0 : BLUE n1 : GREEN n2 : RED'
    # cv2.createTrackbar(switch1, 'image', 1, 2, nothing)
    # switch2 = '0 : BLUE n1 : GREEN n2 : RED n3 : NONE'
    # cv2.createTrackbar(switch2, 'image', 3, 3, nothing)
    # switch3 = '0 : DARK n1 : LIGHT'
    # cv2.createTrackbar(switch3, 'image', 0, 1, nothing)

        # exp = cv2.getTrackbarPos('exponent', 'image')
    exp = 1 + exp/100 # converting exponent to range 1-2
    s1 = int(request.form['s1'])
    s2 = int(request.form['s2'])
    s3 = int(request.form['s3'])
    res = img.copy()
    for i in range(3):
        if i in (s1, s2): # if channel is present
            res[:, :, i] = exponential_function(res[:, :, i], exp) # increasing the values if channel selected
        else:
            if s3: # for light
                res[:, :, i] = exponential_function(res[:, :, i], 2 - exp) # reducing value to make the channels light
            else: # for dark
                res[:, :, i] = 0 # converting the whole channel to 0

    return getPathFile(res, request)



@app.route('/brightness', methods=['POST'])
def brightness():
    imageFile = request.files.get('file', '')
    val = request.form['val']
    img = uploadFileImage(imageFile,request)
    val=int(val)
    # cv2.namedWindow('image')
    # cv2.createTrackbar('val', 'image', 100, 150)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    val = val/100 # dividing by 100 to get in range 0-1.5
    # scale pixel values up or down for channel 1(Saturation)
    hsv[:, :, 1] = hsv[:, :, 1] * val
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255 # setting values > 255 to 255.
    # scale pixel values up or down for channel 2(Value)
    hsv[:, :, 2] = hsv[:, :, 2] * val
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 # setting values > 255 to 255.
    hsv = np.array(hsv, dtype=np.uint8)
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return getPathFile(res, request)




@app.route('/emboss', methods=['POST'])
def emboss():
    imageFile = request.files.get('file', '')
    size = request.form['size']
    switch = request.form['switch']
    img = uploadFileImage(imageFile,request)
    size = int(size)

    # switch = '0 : BL n1 : BR n2 : TR n3 : BR'

    size += 2 # adding 2 to kernel as it a size of 2 is the minimum required.
    s =int(switch)
    height, width = img.shape[:2]
    y = np.ones((height, width), np.uint8) * 128
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = kernel_generator(size) # generating kernel for bottom left kernel
    kernel = np.rot90(kernel, s) # switching kernel according to direction
    res = cv2.add(cv2.filter2D(gray, -1, kernel), y)
    return getPathFile(res, request)



@app.route('/sharp/<type>', methods=['POST'])
def sharp():
    imageFile = request.files.get('file', '')
    data = request.get_json()
    # print(type)
    # return type;
    imageFile.save(os.path.join("uploads", imageFile.filename))

    img = cv2.imread(os.path.join("uploads", imageFile.filename))

    gaussian = np.array(([1/256, 4/256, 6/256, 4/256, 1/256],
                         [4/256, 16/256, 24/256, 16/256, 4/256],
                         [6/256, 24/256, 36/256, 24/256, 6/256],
                         [4/256, 16/256, 24/256, 16/256, 4/256],
                         [1/256, 4/256, 6/256, 4/256, 1/256]), dtype="float")
    blur = np.array(([1/16, 2/16, 1/16],
                     [2/16, 4/16, 2/16],
                     [1/16, 2/16, 1/16]), dtype="float")
    sharp = np.array(([0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]))
    embossing = np.array(([-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]))

    arr = {"gaussian": gaussian, "sharp": sharp, "embossing": embossing, "blur": blur}
    im = cv2.filter2D(img, -1, arr.get(type))

    filename = 'assets/'+'savedImage'+str(random.randint(0, 9999))+'.jpg'
    cv2.imwrite(filename, im)
    path = request.url_root+filename
    print(path)
    return path

@app.route('/classification', methods=['POST'])
def classification():
    imageFile = request.files.get('file', '')
    imageFile.save(os.path.join("uploads", imageFile.filename))
    # img_path = 'meo1.jpg';

    p = predict_label(os.path.join("uploads", imageFile.filename))
    return str(p)

@app.route('/tv-60', methods=['POST'])
def apiTV60():
    imageFile = request.files.get('file', '')
    img = uploadFileImage(imageFile, request)
    # data = request.get_json()
    thresh=request.form['thresh']
    val= request.form['val']
    rs_img_gray = tv_60(img, val, thresh)

    filename = 'assets/' + 'savedImage' + str(random.randint(0, 9999)) + '.jpg'
    cv2.imwrite(filename, rs_img_gray)
    path = request.url_root + filename
    return path


@app.route('/sepia', methods=['POST'])
def sepia():
    imageFile = request.files.get('file', '')
    img = uploadFileImage(imageFile,request)
    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # converting to RGB as sepia matrix is for RGB
    res = np.array(res, dtype=np.float64)
    res = cv2.transform(res, np.matrix([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]]))
    res[np.where(res > 255)] = 255 # clipping values greater than 255 to 255
    res = np.array(res, dtype=np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    return getPathFile(res,request)

@app.route('/negative', methods=['POST'])
def negative():
    imageFile = request.files.get('file', '')
    img = uploadFileImage(imageFile,request)
    res = img.copy()
    res = ~res
    return getPathFile(res,request)


@app.route('/upload', methods=['POST'])
def upload():
    imageFile = request.files.get('file','')
    image = Image.open(imageFile)
    # text = pytesseract.image_to_string(image)
    # f = open("sample.txt", "a")
    # f.truncate(0)
    # f.write(text)
    # f.close()
    # return text

if __name__ == "__main__":
    app.run(debug=True)
