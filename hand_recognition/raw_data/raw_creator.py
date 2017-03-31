import os
import cv2
import numpy


res = numpy.zeros((1, 2701))
classes = numpy.zeros((1, 1))
cl = 0
for folder in xrange(6):
    for file in os.listdir("./"+str(folder)+"/"):

        if file.find("jpg") == -1:
            continue
        path = "./"+str(folder)+"/"+file
        print path, folder
        img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # print "img", img.shape
        img = img.reshape((1, img.shape[0]*img.shape[1]))
        cc = numpy.zeros((1, 1))
        cc[0, 0] = folder
        img = numpy.hstack([cc, img])
        res = numpy.vstack([res, img])

numpy.savetxt("data.data", res, fmt="%d")
numpy.savetxt("class.data", classes, fmt="%d")
