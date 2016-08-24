from collections import deque
import argparse
import imutils
import cv2
import numpy as np
import math
from array import array
from pygame.mixer import Sound, get_init, pre_init
import pygame
import wave
import struct
import win32api

camera = cv2.VideoCapture(0)

while True:
    # last place x, y , w, h
    # grab the current frame
    # (grabbed, frame) = camera.read()
    frame = np.zeros((400, 800, 3))
    print frame.shape
    cv2.circle(frame, (50, 100), 3,
               (0, 255, 255), 2)
    cv2.imshow('asd', frame)
    key = cv2.waitKey(1) & 0xFF  # ?

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break  # cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

