import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

centerX = image.shape[1] // 2
centerY = image.shape[0] // 2

M = cv2.getRotationMatrix2D( (centerX, centerY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Rotated by 45 degrees', rotated)

M = cv2.getRotationMatrix2D( (centerX, centerY), -180, 1.0)
rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Rotated by 180 degrees', rotated)

cv2.waitKey(0)