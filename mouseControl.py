from collections import deque
import argparse
import imutils
import cv2
import numpy
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


def create_noise(frequency=440, amplitude=32600, duration=0.2):
    global counter
    name = 'noise'+str(counter)+".wav"
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

    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask_black_white = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask_black_white, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]  # ?
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)  # ?
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        try:
            xc, yc = center
            xx, yy = 1366, 768

            xf, yf, _ = frame.shape
            xp = int(xx * float(xc)/xf)
            yp = int(yy * float(yc)/yf)
            
            win32api.SetCursorPos((xp, yp))
        except:
            pass
        # update the points queue

        # !!! center can be just compared with old one, and change amplitude or frequency
        # loop over the set of tracked points
        ''' for i in xrange(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            #thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            thickness = 1
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            # show the frame to our screen '''
        cv2.imshow("Frame", mask)  # ? frame , takes to much time!
        key = cv2.waitKey(1) & 0xFF  # ?

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break  # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
