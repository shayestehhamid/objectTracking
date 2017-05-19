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




start = time.time()
results = [[] for _ in range(5)]


for folder in xrange(0, 5):
	size = 0 
	for f in os.listdir("./"+str(folder)+"/"):

		image = cv2.imread("./"+str(folder)+"/"+f, cv2.IMREAD_GRAYSCALE)
		# print image.shape	 (138, 103)

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
		from copy import copy
		kept_image = copy(image)
		# image[image.shape[0]-1, :] = 0
		image = image[0:image.shape[0]-3, :]
		row_size = np.sum(image[image.shape[0]-1, :])

		while True:
			this_row = np.sum(image[image.shape[0]-1, :])
			if this_row >= row_size*.92:
				row_size = this_row
				image  = image[0:image.shape[0]-2, :]
			else:
				break
		image = cv2.dilate(image, None, iterations=2)
		for i in xrange(image.shape[0]):
			for j in xrange(image.shape[1]):
				if image[i, j] > 90:
					image[i, j] = 255
				else:
					image[i, j] = 0
		cv2.imwrite("./res/"+str(folder)+"/"+f, image)
		seen = []
		starts = []
		for i in xrange(image.shape[0]):
			for j in xrange(1, image.shape[1]-1):
				if (i, j) in seen:
					continue
				bfs = []
				if image[i, j] == 255 and (i, j) not in seen:
					starts.append((i, j))
					# print i, j, "this is a start"
					
					cv2.imshow("s", image)
					bfs.append((i, j))

				while(len(bfs)):
					x, y = bfs.pop(0) # i, j
					if (x, y) in seen:
						continue
					
					if image[x, y] == 255:
						cv2.imshow("s", image)
					else:
						continue
					seen.append((x, y))
					image[x, y] = 0
					for _ in [+1, 0]:
						for _a in [-1, 0, 1]:
							newx =  x + _
							newy = y + _a
							if  newx > -1 and newy > -1 and newx < image.shape[0] and newy < image.shape[1]:
								bfs.append((newx, newy)) 
		
		for i, j in starts:
			image[i, j] = 120
		predict = len(starts)
		row1 = -1
		row2 = -1
		if predict == 2:
			row1 = starts[0][0]
			row2 = starts[1][0]
			if abs(row1 - row2) > 15:
				predict = 1
			
		if predict == 1:
			# print image.shape
			image = kept_image
			imgval = []
			for c in xrange(image.shape[1]):
				for r in xrange(image.shape[0]):
					if image[r, c] == 255:
						imgval.append(r)
						break
		
			try:
				if abs(max(imgval) - min(imgval)) < 40:
					predict = 0
			except:
				# print "exception", folder
				predict = 0

		# print predict, folder, f, abs(row1 - row2), row1, row2
		if predict != folder:
			print folder, f, predict
		results[folder].append(predict == folder)

		# cv2.imwrite("./res/"+str(folder)+"/res"+f, image)

finish = time.time()

for i in range(5):
	print i, ": ", float(sum(results[i]))/len(results[i]), sum(results[i]), len(results[i])

print "time", finish - start