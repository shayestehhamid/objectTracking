from collections import deque
import argparse
import imutils
import cv2
import numpy
import math
from array import array
# from pygame.mixer import Sound, get_init, pre_init
# import pygame
import wave
import struct

global counter
counter = 0
global rate
rate = 44100


def create_noise(frequency=440, amplitude=32676):
    global counter
    name = 'noise'+str(counter)+".wav"
    counter += 1
    counter %= 100
    noise_output = wave.open(name, 'w')
    noise_output.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))

    values = []
    rate = 44100
    samples = [float(math.sin(2.0 * math.pi * frequency * t / rate)) for t in xrange(0, int(0.2 * rate))]

    for i in range(0, len(samples)):
        packed_value = struct.pack('h', int(amplitude * samples[i]))

        values.append(packed_value)
        values.append(packed_value)

    value_str = ''.join(values)
    noise_output.writeframes(value_str)
    noise_output.close()
    return name

# def init_sound_player():
#     pygame.mixer.pre_init()
#     pygame.mixer.init()
#     channel = pygame.mixer.Channel(0)
#     return channel


def main():
    global rate
    # channel = init_sound_player()


    frequency = 440
    amplitude = 32600
    rate = 44100
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=15,
                    help="max buffer size")
    args = vars(ap.parse_args())
    # print args
    # print type(args)

    color_lower = (50, 50, 50)
    color_upper = (255, 255, 255)
    pts = deque(maxlen=args["buffer"])
    dps = deque(maxlen=args["buffer"])

    if args["video"] is None:
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
    last_hit = False
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()


        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame2 = imutils.resize(frame, width=400)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        black_whit_mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(black_whit_mask, None, iterations=5)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]  # ?
        center = None
        radius = 0
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #print radius
            M = cv2.moments(c)  # ?
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame2, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.line(frame, (0, 225), (600, 225), (0, 255, 255), thickness=2)
        cv2.line(frame, (200, 0), (200, 450), (0, 255, 255), thickness=2)
        cv2.line(frame, (400, 0), (400, 450), (0, 255, 255), thickness=2)
        # update the points queue
        depth = radius / 233
        print depth
        flag = False
        if center is not None:
            pts.appendleft(center)
            dps.appendleft(depth)
            flag = True

        if len(pts) > 1 and flag:
            change_x = pts[-1][0] - pts[-2][0] if (pts[-1] and pts[-2]) else 0
            # change_y = pts[-1][1] - pts[-2][1] if (pts[-1] and pts[-2]) else 0

            frequency += change_x * 10
            frequency = (abs(frequency) % 60000) + 47 if change_x else frequency
            # amplitude += change_y * 25
            # amplitude = abs(amplitude) % 32600 if change_y else amplitude
            # amplitude = depth * 32600

            amplitude = 0
            hit = False
            if dps[-2] > .35 and dps[-1] < dps[-2] and not last_hit:
                ds = -2
                hit = True
                last_hit = True
                sized = 0
                while dps[ds] >= dps[ds - 1] and sized < 2:
                    ds -= 1
                    sized += 1

                amplitude = int(((dps[-2] - dps[ds]) / float(3)) * 32600)
                # print dps, dps[-2] - dps[ds], ds, dps[-1]
                # print amplitude

            if dps[-2] <= .35:
                last_hit = False

            # place = int(pts[-1][1] > 225) * 3 + int(pts[-1][0]/200)
            # print pts[-1]
            # print place
            # noise_file = create_noise(frequency, amplitude)
            # pygame.mixer.music.load(noise_file)

            # if hit:
            #     pygame.mixer.music.play()


            # print frequency, amplitude, noise_file

        # !!! center can be just compared with old one, and change domain or frequency
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
        cv2.imshow("Frame", mask)  # ?
        cv2.imshow('original', frame)
        cv2.imshow('noise', black_whit_mask)
        cv2.imshow('winname', frame2)
        global counter
        counter += 1
        cv2.imwrite(str(counter) + ".jpg", frame2)        
        key = cv2.waitKey(1) & 0xFF  # ?

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break  # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()