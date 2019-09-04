import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

(b, g, r) = image[0, 0]
print('Pixel at (0,0) - Red {}, Green {}, Blue {}'.format(r, g, b))

image[100:200, 100:200] = (100, 0, 0)

cv2.imshow("Updated", image)
cv2.waitKey(0)