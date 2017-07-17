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



global counter
counter = 0
global frequency_range
frequency_range = 16050
global rate
rate = 44100


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
        import copy
        #yc = copy.copy(frame)
        firstframe = copy.copy(frame)
        yc = cv2.cvtColor(frame, cv2.COLOR_RGB2YCR_CB)

        cv2.equalizeHist(yc[0])
        frame = cv2.cvtColor(yc, cv2.COLOR_YCR_CB2RGB)

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

        cv2.imshow('frame', temp_frame)
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


        left_side = max(x - 30 - R, 0)
        right_side = x + 30 + R
        up_side = max(y - 30 - R, 0)
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
        white_frame = frame[up_side + 20:down_side - 35, left_side + 20:right_side - 30, :]
        frame = frame[up_side:down_side, left_side:right_side, :]

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        blue, red, green = cv2.split(frame)

        red_white_channel, blue_white_channel, green_white_channel = cv2.split(white_frame)

        red_erode = cv2.erode(red_white_channel, None, iterations=1)
        red_mask = cv2.dilate(red_erode, None, iterations=1)
        ret, red_white = cv2.threshold(red_mask, 100, 255, cv2.THRESH_BINARY)

        blue_erode = cv2.erode(blue_white_channel, None, iterations=1)
        blue_mask = cv2.dilate(blue_erode, None, iterations=1)
        ret, blue_white = cv2.threshold(blue_mask, 100, 255, cv2.THRESH_BINARY)

        green_erode = cv2.erode(green_white_channel, None, iterations=1)
        green_mask = cv2.dilate(green_erode, None, iterations=1)
        ret, green_white = cv2.threshold(green_mask, 100, 255, cv2.THRESH_BINARY)

        white_mask = np.multiply(np.multiply(green_white, red_white), blue_white)
        # print sum(sum(white_mask))
        diff_green = green - np.maximum(red, blue)
        green_erode = cv2.erode(diff_green, None, iterations=3)
        green_mask = cv2.dilate(green_erode, None, iterations=1)
        ret, green_mask = cv2.threshold(green_mask, 120, 255, cv2.THRESH_BINARY)

        mask = green_mask  # decide on color
        cnts_temp = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]  # ?

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
        cv2.circle(keep_frame, (int(CR[0][0]), int(CR[0][1])), int(CR[1]), (255, 0, 0), 3)
        def size(p):
            return (sum([x*x for x in p]))**(0.5)

        def tetha(CR):
            import math
            sc = size((CR[0] - 320, CR[1]-240))
            sa = size((-320, 0))
            dot = (CR[0]-320)*(-320)
            cost = float(dot)/(sc*sa)
            # print CR, sc, sa, dot, cost
            return math.degrees(math.acos(cost))

        angle = tetha(CR[0])
        if CR[0][1] > 240:
            angle = 360 - angle
        cell = int(angle/45)
        if size((CR[0][0] - 320, CR[0][1]-240)) < 123:
            cell = 8

        print cell

        if flag:
            try:
                pass
                # old_pos = (x, y)
                #
                # new_pos = (CR[0][0], CR[0][1])
                # x_dis, y_dis = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]
                #
                # dis = (x_dis**2 + y_dis**2)**.5
                dis = 0
                if dis < 3:
                    pass
                else:
                    # xp, yp = win32api.GetCursorPos()

                    # xx, yy = 1366, 768
                    # xp += x_dis
                    # yp += y_dis
                    # xp = max(0, min(int(xp), xx))
                    # yp = max(0, min(int(yp), yy))
                    pass
                    # win32api.SetCursorPos((xp, yp))

                ss = sum(sum(white_mask))
                # print "ss: ", ss
                # if ss > 9000:
                #     noise_file = create_noise()
                #     print noise_file
                #     pygame.mixer.music.load(noise_file)
                #     pygame.mixer.music.play()

                    # click on xp, yp

            except:
                pass
            # update the points queue


        cv2.circle(keep_frame, center=(320, 240), radius=120, color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(0, 240), color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(0, 480), color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(0, 0), color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(320, 0), color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(640, 0), color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(640, 480), color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(640, 240), color=(0, 255, 255), thickness=3)
        cv2.line(keep_frame, pt1=(320, 240), pt2=(320, 480), color=(0, 255, 255), thickness=3)
        cv2.circle(keep_frame, center=(0, 240), radius=5, color=(255, 0, 0), thickness=3)
        # cv2.imshow("frame", frame)
        # cv2.imshow('frame2', green_mask)
        cv2.imshow('lcal', frame)
        cv2.imshow('frame3', keep_frame)
        cv2.imshow('frame4', white_mask)

        key = cv2.waitKey(10) & 0xFF  # ?

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break  # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
