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

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

if args["video"] is None:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    # last place x, y , w, h
    # grab the current frame
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)

    # print "left ", left_side, "right: ", right_side, "up: ", up_side, "down: ", down_side
    # print keep_frame.shape

    # frame = frame[up_side:down_side, left_side:right_side, :]
    # cropped = frame.copy()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    blue, red, green = cv2.split(frame)

    red_erode = cv2.erode(red, None, iterations=3)
    red_mask = cv2.dilate(red_erode, None, iterations=1)
    ret, red_white = cv2.threshold(red_mask, 120, 255, cv2.THRESH_BINARY)

    blue_erode = cv2.erode(blue, None, iterations=3)
    blue_mask = cv2.dilate(red_erode, None, iterations=1)
    ret, blue_white = cv2.threshold(blue_mask, 120, 255, cv2.THRESH_BINARY)

    green_erode = cv2.erode(green, None, iterations=3)
    green_mask = cv2.dilate(green_erode, None, iterations=1)
    ret, green_white = cv2.threshold(green_mask, 120, 255, cv2.THRESH_BINARY)

    mask = np.multiply(np.multiply(green_white, red_white), blue_white)
    cnts_temp = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[-2]  # ?
    center = None
    ind = 0
    cnts = []

    for c in cnts_temp:
        # print cv2.contourArea(c), max(1, last_place[2]-15)*max(1, last_place[3]-15), \
        #     max(1, last_place[2]+15) * max(1, last_place[3]+15)
        # if (max(1, last_place[2]-15))*(max(1, last_place[3]-15)) < cv2.contourArea(c) \
        #         < (max(1, last_place[2]+15))*(max(1, last_place[3]+15)):
        if 150 < cv2.contourArea(c) < 3000:
            cnts.append(c)

    # only proceed if at least one contour was found
    # if len(cnts) > 0:
    center = 0
    flag = False
    # if not len(cnts):
    #     continue
    while len(cnts):
        break
        c = max(cnts, key=cv2.contourArea)
        # check_value = check_rec(c, green_mask)
        # if check_value > -1:
        # print check_value, cv2.contourArea(c)
        ((xc, yc), radius) = cv2.minEnclosingCircle(c)
        if radius > 14 and False:

            # CR = ((xc + left_side, yc + up_side), radius)

            # print green_mask[x:x+w, y:y+h]
            # center = (x + left_side + w / 2, y + up_side + h / 2)

            cv2.circle(frame, (int(xc), int(yc)), int(radius), (0, 255, 0))  # draw rectangle!
            flag = True
            break
        else:
            cnts.remove(c)

    # ? frame , takes to much time!
    # cv2.circle(keep_frame, (600, 0), 10, (255, 0, 0))
    cv2.imshow("frame", frame)
    cv2.imshow('frame2', mask)
    # cv2.imshow('frame3', keep_frame)

    key = cv2.waitKey(10) & 0xFF  # ?

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break  # cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
