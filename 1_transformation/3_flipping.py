import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow("Original", image)

flipped = cv2.flip(image, 1)
cv2.imshow("Horizontally", flipped)

flipped = cv2.flip(image, 0)
cv2.imshow("Vertically", flipped)

flipped = cv2.flip(image, -1)
cv2.imshow("Horizontally & Vertically", flipped)

cv2.waitKey(0)