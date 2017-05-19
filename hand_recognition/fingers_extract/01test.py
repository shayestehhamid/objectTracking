from collections import deque
import argparse
import imutils
import cv2
import scipy
import numpy as np
import urllib
import time
import math
import os

vals = [[], []]
 

for folder in xrange(0, 2):
	size = 0 

	for f in os.listdir("./"+str(folder)+"/"):
		image = cv2.imread("./"+str(folder)+"/"+f, cv2.IMREAD_GRAYSCALE)
		left, right, up = -1, -1, -1
		for i in xrange(image.shape[1]):
			s = np.sum(image[:, i])
			if s > 5*255:
				left = i
				break
		for i in xrange(image.shape[1]-1, 0, -1):
			s  = np.sum(image[:, i])
			if s > 5*255:
				right = i
				break
		for i in xrange(image.shape[0]-1):
			s = np.sum(image[i, :])
			if s > 5*255:
				up = i
				break
		image = image[up:, left:right]

		imgval = []
		for c in xrange(image.shape[1]):
			for r in xrange(image.shape[0]):
				if image[r, c] == 255:
					imgval.append(r)
					break
		vals[folder].append((max(imgval), min(imgval)))

for c in vals:
	m =  0
	for i in c:
		m  += i[0] - i[1]
	print float(m)/len(c), m, len(c)
	print 20*"*"