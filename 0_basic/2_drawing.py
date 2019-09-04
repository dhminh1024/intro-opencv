import numpy as np
import cv2

canvas = np.zeros((300, 300, 3), dtype="uint8")

green = (0, 255, 0)
cv2.line(canvas, (0,0), (300,300), green)

red = (0, 0, 255)
cv2.line(canvas, (300,0), (0,300), red, 3)

cv2.rectangle(canvas, (50,200), (200,225), red, 3)

cv2.rectangle(canvas, (200,50), (225,125), green, -1)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)