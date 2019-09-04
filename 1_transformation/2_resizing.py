import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow("Original", image)

r = image.shape[0] / image.shape[1]
dim = (150, int(150 * r))

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized",resized)

cv2.waitKey(0)
