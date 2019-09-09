import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from skimage.feature import hog

# load model
model = joblib.load("digits_cls.pkl")

# read image
img = cv2.imread('digit-reco-1-in.jpg')
img = cv2.resize(img, (400, int(img.shape[0]/img.shape[1]*400)))

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,3), 0)

# find contours
(ret, thres1) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
(ret, thres3) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
thres2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
thres = cv2.bitwise_and(thres1, thres2, thres3)
cann = cv2.Canny(thres, 200, 255)
contours, hie = cv2.findContours(cann.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# decode operations
operation_decode = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:'+', 11:'-', 12:'x', 13:'/'}

# arrange contours from left to right
for i in range(len(contours) - 1):
    for j in range(len(contours) - 1):
        if contours[j+1][0][0][0] < contours[j][0][0][0]:
            d = contours[j+1]
            contours[j+1] = contours[j]
            contours[j] = d

conn = []
i = 0
while i < len(contours) - 1:
    x, y, w, h = np.min(contours[i][:, 0][:, 0]), np.min(contours[i][:, 0][:, 1]),np.max(contours[i][:, 0][:, 0]), np.max(contours[i][:, 0][:, 1])
    print(x,y,w,h)
    j = i+1
    if w + 10 > np.min(contours[j][:, 0][:, 0]):
        while ((w > np.min(contours[j][:, 0][:, 0])) & (j < len(contours)-1)):
            x1, y1, w1, h1 = np.min(contours[j][:, 0][:, 0]), np.min(contours[j][:, 0][:, 1]),np.max(contours[j][:, 0][:, 0]), np.max(contours[j][:, 0][:, 1])
            x = np.min([x, x1])
            y = np.min([y, y1])
            w = np.max([w, w1])
            h = np.max([h, h1])
            j += 1  
    conn.append([x,y,w,h])
    i = j
print(conn, 'conn1')

x, y, w, h = np.min(contours[-1][:, 0][:, 0]), np.min(contours[-1][:, 0][:, 1]),np.max(contours[-1][:, 0][:, 0]), np.max(contours[-1][:, 0][:, 1])
print(x,y,w,h, 'conn2')
if conn[-1][2] + 2 > x:
    x = np.min([conn[-1][0], x])
    y = np.min([conn[-1][1], y])
    w = np.max([conn[-1][2], w])
    h = np.max([conn[-1][3], h])
    j = len(conn) - 1
    while conn[j-1][2] + 2 > conn[j][0]:
        x = np.min([conn[j][0], conn[j-1][0]])
        y = np.min([conn[j][1], conn[j-1][1]])
        w = np.max([conn[j][2], conn[j-1][2]])
        h = np.max([conn[j][3], conn[j-1][3]])
        conn[j-1] = [x,y,w,h]
        conn.pop()
        j -= 1
    else:
        conn[-1] = [x,y,w,h]
else:
    conn.append([x,y,w,h])

mean_X = np.mean(conn, 1)[0]
mean_Y = np.mean(conn, 1)[1]

# extract all digits and operations and store to all_digits
all_digits = []
for i in range(len(conn)):
    # Find the bounding box of digits and operations
    x, y, w, h = conn[i] 
    # case 1: minus operation
    if (w-x) > 3*(h-y):
        x_new = x 
        y_new = y - (w-x)//2
        w_new = w 
        h_new = h + (w-x)//2  
    # case 2: add and multiple operations
    elif (h-y < 6/7*mean_Y):
        x_new = x 
        y_new = y 
        w_new = w 
        h_new = h             
    # case 3: divide operation and digits
    else:
        x_new = x - (w-x)//10
        y_new = y - (h-y)//10
        w_new = w + (w-x)//10
        h_new = h + (h-y)//10

    # ensure no trash contour!
    if (((x_new!=w_new) | (y_new!=h_new)) & (w-x > 1/5*mean_X)):

        # make the rectangle area around the digit
        cv2.rectangle(img, (x_new,y_new), (w_new,h_new), (255,0,0), 2)

        # getting out digits
        digit = thres[y_new:h_new, x_new:w_new]
        # cv2.imshow('gray', digit)
        # cv2.waitKey(0)
        digit = cv2.resize(digit, (28,28), interpolation=cv2.INTER_AREA)
        digit = cv2.dilate(digit, (3, 3))
        digit_hog_fd = hog(digit, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        digit_label = model.predict(digit_hog_fd.reshape(1,-1))
        cv2.putText(img, str(operation_decode[int(digit_label[0])]), (x, y), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3)  
        all_digits.append(str(digit_label[0]))

# show image with bounding box
cv2.imshow('gray', img)
cv2.waitKey(0)

# first we do the multiple and divide
s = []
i = 0
print(all_digits)
while i < len(all_digits) - 1:
    if all_digits[i + 1] == '12':
        m = int(all_digits[i])*int(all_digits[i+2])
        s.append(str(m))
        i += 3
    elif all_digits[i + 1] == '13':
        m = int(all_digits[i])/int(all_digits[i+2])
        s.append(str(m))
        i += 3
    else:
        s.append(all_digits[i])  
        i += 1

if all_digits[-2] not in ['12', '13']:
    s.append(all_digits[-1])

# Then we do the add and minus
result = 0
if s[0] not in ['10', '11']:
    result += float(s[0])
for i in range(1, len(s)-1):
    if s[i] == '10':
        result += float(s[i+1])
    elif s[i] == '11':
        result -= float(s[i+1])

print(result)

cv2.imshow('gray', img)
cv2.waitKey(0)