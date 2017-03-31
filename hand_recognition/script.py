import cv2
import numpy as np

img = cv2.imread("hand.jpg")
colors = np.zeros((1, 3))
cn = 0
for i in xrange(img.shape[0]):
    for j in xrange(img.shape[1]):
        colors = np.add(colors, img[i, j])
        cn += 1

print colors/float(cn)