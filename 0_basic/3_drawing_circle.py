import numpy as np
import cv2

canvas = np.zeros((300, 300, 3), dtype="uint8")
white = (255, 255, 255)
red = (0, 0, 180)
dark_blue = (100, 0, 0)
centerX = canvas.shape[1] // 2
centerY = canvas.shape[0] // 2

cv2.circle(canvas, (centerX, centerY), 150, red, -1)
cv2.circle(canvas, (centerX, centerY), 120, white, -1)
cv2.circle(canvas, (centerX, centerY), 90, red, -1)
cv2.circle(canvas, (centerX, centerY), 60, dark_blue, -1)


# for radius in range(150, 49, 25):
#     cv2.circle(canvas, (centerX, centerY), radius, white, -1)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)