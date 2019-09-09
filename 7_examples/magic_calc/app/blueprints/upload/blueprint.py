from flask import Blueprint, Flask, request, redirect, jsonify, current_app
import re
import base64
import os
import numpy as np
import cv2
import tensorflow as tf
import mahotas
import joblib
from sklearn.externals import joblib
from skimage.feature import hog
import imutils

upload = Blueprint('upload', __name__)

model_loaded = False
model_digits = None
model_operators = None
operator_dict = {
  0 : '+',
  1 : '-',
  2 : '/',
  3: '*'
}

def deskew(image, width):
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5*w*skew],
                    [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    image = imutils.resize(image, width=width)

    return image

def center_extent(image, size):
    (eW, eH) = size

    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=eW)
    else:
        image = imutils.resize(image, height=eH)

    extent = np.zeros((eH, eW), dtype='uint8')
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX+image.shape[1]] = image

    CM = mahotas.center_of_mass(extent)
    (cY, cX) = np.round(CM).astype("int32")
    (dX, dY) = ((size[0]//2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)

    return extent

def load_model():
  global model_loaded, model_digits, model_operators
  # model_num = tf.keras.models.load_model("static/models/" + MODEL_NUM)
  # Load the classifier
  model_digits_operators = joblib.load("static/models/digits_operator_cls.pkl")
  model_digits_svm = joblib.load("static/models/digits_cls.pkl")
  model_operators_svm = joblib.load("static/models/operators_cls.pkl")
  model_digits_cnn = tf.keras.models.load_model("static/models/handwritten_model.h5")
  model_digits_ops_cnn = tf.keras.models.load_model("static/models/mnist_operators_cnn.h5")
  model_operators_cnn = tf.keras.models.load_model("static/models/operators_cnn.h5")
  model_operators_cnn_minhdh = tf.keras.models.load_model("static/models/minh_operator.h5")

  model_loaded = True
  model_digits = model_digits_cnn
  model_operators = model_operators_svm

def processing(image, mode):
    CROPPED_W = 554
    CROPPED_H = 166

    if mode == 'camera':
        # image ratio w/h
        im_ratio = round(image.shape[1]/image.shape[0], 2)
        center_w = image.shape[1]//2
        center_h = image.shape[0]//2
        x = center_w-(CROPPED_W//2)
        y = center_h-(CROPPED_H//2)

        # crop the image
        image = image[y:y + CROPPED_H, x:x + CROPPED_W]
    
    # image is cropped
    #resize
    img = cv2.resize(image, (400, int(image.shape[0]/image.shape[1]*400)))

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # find contours
    (ret, thres1) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    (ret, thres3) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thres2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
    edge = cv2.bitwise_and(thres1, thres2, thres3)
    # edge = cv2.Canny(edge, 200, 255)
    contours, hie = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite('uploads/test.jpg',edge)

    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in  contours], key=lambda x: x[1])

    equation = []
    rects = []

    for (i, (c, _) ) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if h > 5 and w > 5:
            rects.append((x, y, w, h))
            roi = edge[y:y+int(1.2*h), x:x+w]
            thresh = roi.copy()

            thresh = deskew(thresh, 28)
            thresh = center_extent(thresh, (28, 28))
            # thresh = np.reshape(thresh, (28, 28, 1))
            thresh = thresh / 255

            if i%2==1: # operator
                roi_hog_fd = hog(thresh, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
                prediction = model_operators.predict(np.array([roi_hog_fd], 'float64'))[0]
                prediction = operator_dict[prediction]
            else: # digit
                prediction = np.argmax(model_digits.predict(np.expand_dims(thresh, axis=0)), axis=1)[0]
                prediction = str(prediction)

            equation.append(prediction)

    calculator = ''.join(str(item) for item in equation)
    equation = (calculator + ' = ' + str(eval(calculator)))
    results = {'image':'', 'status':1, 'equation':equation}
    return results
  
    

@upload.route('/upload/', methods=['POST'])
def handle_upload():
    if request.method == 'POST':
        global model_loaded, model_digits, model_operators

        if not model_loaded:
            load_model()

        data = request.get_json()

        # Preprocess the upload image
        img_raw = data['data-uri'].encode()
        img_str = re.search(b"base64,(.*)", img_raw).group(1)
        img_decode = base64.decodebytes(img_str)

        # # Write the image to the server
        # upload file to storage
        # filename = 'output.jpg'
        # with open(os.path.join(current_app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
        #   f.write(img_decode)

        # Prediction
        nparr = np.fromstring(img_decode, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = processing(image, data['mode'])

        # if results['status'] == 1:
        # with open(results['calculated_img'], "rb") as image_file:
        #     encoded_string = base64.b64encode(image_file.read())
        #     results['image'] = encoded_string.decode()

        return jsonify(results)