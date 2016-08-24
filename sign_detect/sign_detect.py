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


global counter
counter = 0
global frequency_range
frequency_range = 16050
global rate
rate = 44100


def check_rec(c, frame):
    x, y, w, h = cv2.boundingRect(c)
    area = w * h

    sp = frame[x:x+w, y:y+h]
    ones = np.ones(sp.shape)
    # print frame[x:x+w, y:y+h]
    # print frame
    try:
        real_area = float(sum(sum(np.multiply(sp, ones))))/255
        return (area - float(real_area))/area
    except:
        return 0


def show(frame):
    cv2.imshow("Frame", frame)  # ? frame , takes to much time!


def initial_sign_detection(camera, args):
    place = (-1, -1)
    # print "start"
    while True:
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=400)
        frame = cv2.flip(frame, 1)
        cv2.imshow("ff", frame)
        if not grabbed:
            print "nemishe"
            continue
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blue, red, green = cv2.split(frame)
        # diff_green = green - gray_scale
        diff_green = green - np.maximum(red, blue)
        green_erode = cv2.erode(green, None, iterations=2)
        green_mask = cv2.dilate(green_erode, None, iterations=2)
        ret, green_mask = cv2.threshold(green_mask, 120, 255, cv2.THRESH_BINARY)
        #
        # green_mask = green_mask / 255
        # red_mask = red_mask / 255
        # blue_mask = blue_mask / 255
        #
        # green_mask, red_mask, blue_mask = green_mask - (red_mask + blue_mask), \
        #                                   red_mask - (green_mask + blue_mask), blue_mask - (green_mask + red_mask)
        # green_mask = ((green_mask + 3) / 4) * 255
        # red_mask = ((red_mask + 3) / 4) * 255
        # blue_mask = ((blue_mask + 3) / 4) * 255
        #
        # red_mask = red_mask - (green_mask + blue_mask)
        #
        # blue_mask = blue_mask - (green_mask + red_mask)

        # print green_mask
        # ret_green, green_mask = cv2.threshold(green_mask, .5, 3, cv2.THRESH_BINARY)
        # print green_mask

        # mask = red_mask  # decide on color
        # cnts_temp = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        #                              cv2.CHAIN_APPROX_SIMPLE)[-2]  # ?
        # center = None
        # ind = 0
        # cnts = []
        # for c in cnts_temp:
        #     if cv2.contourArea(c) > 40:
        #         cnts.append(c)

        # only proceed if at least one contour was found
        # if len(cnts) > 0:
        #     # find the largest contour in the mask, then use
        #     # it to compute the minimum enclosing circle and
        #     # centroid
        #     c = max(cnts, key=cv2.contourArea)
        #     ((x, y), radius) = cv2.minEnclosingCircle(c)
        #     M = cv2.moments(c)  # ?
        #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #     cv2.circle(frame, (int(x), int(y)), int(radius),
        #                (0, 255, 255), 2)
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)


        show(green_mask)


        # cordinate, area,
    # print "done"


def create_noise(frequency=440, amplitude=32600, duration=0.2):
    global counter
    name = 'noise' + str(counter) + ".wav"
    counter += 1
    counter %= 100
    rate = 44100
    noise_output = wave.open(name, 'w')
    noise_output.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))

    values = []

    samples = [float(math.sin(2.0 * math.pi * frequency * t / rate)) for t in xrange(0, int(duration * rate))]

    for i in range(0, len(samples)):
        packed_value = struct.pack('h', int(amplitude * samples[i]))

        values.append(packed_value)
        values.append(packed_value)

    value_str = ''.join(values)
    noise_output.writeframes(value_str)
    noise_output.close()
    return name


def init_sound_player():
    pygame.mixer.pre_init()
    pygame.mixer.init()
    channel = pygame.mixer.Channel(0)
    return channel


def main():
    global rate
    channel = init_sound_player()

    frequency = 440
    amplitude = 32600
    rate = 44100
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())
    # print args
    # print type(args)
    #
    # color_lower = (29, 86, 6)
    # color_upper = (64, 255, 255)
    color_lower = (50, 50, 50)
    color_upper = (255, 255, 255)
    pts = deque(maxlen=args["buffer"])

    if args["video"] is None:
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
    # initial_sign_detection(camera, args)


    # last_places = (x, y, w, h)
    CR = ((0, 0), 0)
    # initializeing phase

    found = False

    while not found:
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break


        frame = cv2.flip(frame, 1)

        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blue, red, green = cv2.split(frame)
        # diff_green = green - gray_scale
        diff_green = green - np.maximum(red, blue)
        green_erode = cv2.erode(diff_green, None, iterations=2)
        green_mask = cv2.dilate(green_erode, None, iterations=1)
        ret, green_mask = cv2.threshold(green_mask, 120, 255, cv2.THRESH_BINARY)

        mask = green_mask  # decide on color
        cnts_temp = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]  # ?
        center = None
        ind = 0
        cnts = []

        for c in cnts_temp:
            if 4000 > cv2.contourArea(c) > 200:
                cnts.append(c)

        center = 0
        if not len(cnts):
            continue

        c = max(cnts, key=cv2.contourArea)
        # check_value = check_rec(c, green_mask)
        # if check_value > -1:
        # print check_value, cv2.contourArea(c)

        temp_frame = frame.copy()
        ((x, y), R) = cv2.minEnclosingCircle(c)
        # print green_mask[x:x+w, y:y+h]
        center = (x, y, R)
        cv2.circle(temp_frame, (int(x), int(y)), int(R), (0, 255, 0))  # draw rectangle!
        show(temp_frame)
        cv2.imshow('as', mask)
        key = cv2.waitKey(1000) & 0xFF  # ?
        if key == ord('y'):
            found = True
            CR = ((x, y), R)
            break

    # print "this is the init phase!"
    # print CR
    ## end of init phase!

    # print camera
    while True:
        # last place x, y , w, h
        # grab the current frame
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        keep_frame = frame.copy()
        x = int(CR[0][0])
        y = int(CR[0][1])
        R = int(CR[1])

        page_size = 3
        # down_side = x-int(page_size * last_place[2])
        # up_side = x+int(page_size * last_place[2])
        # left_side = y-int(page_size * last_place[3])
        # right_side = y+int(page_size * last_place[3])

        left_side = max(x - 30 - R, 0)
        right_side = x + 30+ R
        up_side = max(y - 30- R, 0)
        down_side = y + 30 + R

        for q in [left_side, right_side, up_side, down_side]:
            if q < 0:
                q = 0

        while right_side - left_side < 180:
            right_side = min(right_side + 10, frame.shape[1])
            left_side = max(left_side - 10, 0)

        while down_side - up_side < 180:
            up_side = max(0, up_side-10)
            down_side = min(down_side + 10, frame.shape[0])

        right_side = min(right_side, frame.shape[1])
        down_side = min(down_side, frame.shape[0])
        # print "left ", left_side, "right: ", right_side, "up: ", up_side, "down: ", down_side
        # print keep_frame.shape

        frame = frame[up_side:down_side, left_side:right_side, :]
        cropped = frame.copy()

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        blue, red, green = cv2.split(frame)

        diff_green = green - np.maximum(red, blue)
        green_erode = cv2.erode(diff_green, None, iterations=3)
        green_mask = cv2.dilate(green_erode, None, iterations=1)
        ret, green_mask = cv2.threshold(green_mask, 120, 255, cv2.THRESH_BINARY)

        mask = green_mask  # decide on color
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
            c = max(cnts, key=cv2.contourArea)
            # check_value = check_rec(c, green_mask)
            # if check_value > -1:
                # print check_value, cv2.contourArea(c)
            ((xc, yc), radius) = cv2.minEnclosingCircle(c)
            if radius > 14:

                CR = ((xc + left_side, yc + up_side), radius)

                # print green_mask[x:x+w, y:y+h]
                # center = (x + left_side + w / 2, y + up_side + h / 2)

                cv2.circle(frame, (int(xc), int(yc)), int(radius), (0, 255, 0))  # draw rectangle!
                flag = True
                break
            else:
                cnts.remove(c)

        if flag:
            try:

                old_pos = (x, y)

                new_pos = (CR[0][0], CR[0][1])
                x_dis, y_dis =  new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]

                dis = (x_dis**2 + y_dis**2)**.5
                print 'dis: ', dis
                if dis < 3:
                    pass
                else:
                    xp, yp = win32api.GetCursorPos()
                    print "old: ", old_pos, "new : ", new_pos, "xdis: ", x_dis, "ydis : ", y_dis, "dis: ", dis
                    print "CR: ", CR[0]
                    print "old xp : ", xp, "old yp: ", yp
                    xx, yy = 1366, 768
                    xp += x_dis
                    yp += y_dis
                    xp = max(0, min(int(xp), xx))
                    yp = max(0, min(int(yp), yy))
                    print "new xp : ", xp, "new yp: ", yp
                    win32api.SetCursorPos((xp, yp))

            except:
                pass
            # update the points queue

        # !!! center can be just compared with old one, and change amplitude or frequency
        # loop over the set of tracked points



        # ? frame , takes to much time!
        # cv2.circle(keep_frame, (600, 0), 10, (255, 0, 0))
        cv2.imshow("frame", frame)
        cv2.imshow('frame2', green_mask)
        cv2.imshow('frame3', keep_frame)

        key = cv2.waitKey(10) & 0xFF  # ?

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break  # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
