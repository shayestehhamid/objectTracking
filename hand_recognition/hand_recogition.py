from collections import deque
import argparse
import imutils
import cv2
import scipy
import numpy as np


class Hand_Detect():

    def hand_color_detect(self, frame):
        from copy import copy
        frame = copy(frame)
        # 0 blau, 1 grun, 2 rot
        # return frame[:, :, 2]
        res1 = scipy.less(frame[:, :, 1], frame[:, :, 2])
        res2 = scipy.less(frame[:, :, 0], frame[:, :, 1])
        return (255*scipy.logical_and(res1, res2)).astype(np.uint8)

    def dist(self, frame, prev_frame):
        frame1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        prev_frame1 = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        dist_frame = prev_frame1 - frame1
        mn = np.ndarray.sum(frame1) / float(frame1.shape[0] * frame1.shape[1])
        # print "dist", dist_frame.shape, type(dist_frame[0, 0])
        dist_frame = cv2.threshold(dist_frame, 0.2 * mn, 1000, cv2.THRESH_BINARY)[1]
        # print dist_frame[1]
        dist_frame = cv2.erode(dist_frame, None, iterations=3)
        # print "dist", dist_frame.shape, type(dist_frame)
        # dist_frame = cv2.dilate(dist_frame, None, iterations=2)
        return dist_frame

    def canny(self, frame):
        canny = cv2.Canny(frame, 100, 200)
        canny = cv2.dilate(canny, None, iterations=2)
        return canny


class Hand_Detect_Number(Hand_Detect):

    def __init__(self, frame):
        self.number = 0
        self.prev_frame = frame
        self.frame = frame
        self.edge_frame = None
        self.hand_colored_frame = None
        self.res = None
        self.color_value = np.array([26,   61,  118])
        self.sample_done = False
        from sklearn.lda import LDA
        self.clf = LDA()
        self.background = None
        self.remove_background_frame = frame

    def set_background(self, frame):
        self.background = frame

    def remove_background(self):
        self.remove_background_frame = np.subtract(self.frame, self.background)
        vabs = np.vectorize(abs)
        res = vabs(self.remove_background_frame)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        res = 255 * res
        res.astype(np.uint8)
        self.remove_background_frame = cv2.erode(res, None, iterations=3)
        return self.remove_background_frame

    def hand_color_d(self, frame):
        self.hand_colored_frame = self.hand_color_detect(frame)

        # print type(self.hand_colored_frame[0, 0])
        # print type(self.color_value)
        #
        # temp_value = self.frame - self.color_value
        # print "size", self.color_value.shape
        # x = np.apply_along_axis(np.abs, 2, temp_value)
        # x = np.apply_along_axis(np.max, 2, x)
        # x = x.astype(np.uint8)
        # rr = cv2.threshold(x, 100, 0, cv2.THRESH_BINARY)[1]
        # cv2.imshow("test", rr)
        # print "rr", rr

        return self.hand_colored_frame

    def run(self, frame):
        self.prev_frame = self.frame
        self.frame = frame
        self.dist(self.frame, self.prev_frame)
        self.edge_frame = self.canny(frame)
        self.hand_colored_frame = self.hand_color_detect(frame)
        self.remove_background()
        # self.res = scipy.logical_and(self.hand_colored_frame, self.edge_frame)
        # self.res = scipy.logical_and(self.remove_background_frame, self.res)
        self.res = scipy.logical_and(self.hand_colored_frame, self.remove_background_frame)
        self.res = 255 * self.res
        self.res = self.res.astype(np.uint8)
        self.res = cv2.erode(self.res, None, iterations=2)
        self.res = cv2.dilate(self.res, None, iterations=2)

    def show(self):
        # cv2.imwrite(str(self.number)+".jpg", self.res)
        self.number += 1
        # cv2.imshow("frame", self.frame)
        # cv2.imshow("canny", self.edge_frame)
        # cv2.imshow("color", self.hand_colored_frame)
        cv2.imshow("res", self.res)
        cv2.imshow("background", self.background)
        cv2.imshow("frame", self.frame)
        cv2.imshow("removed background", self.remove_background_frame)


    def train(self, data, cls):
        self.clf.fit(data, cls)

    def predict(self):
        return self.clf.predict(self.res.reshape((1, self.res.shape[0]*self.res.shape[1])))

class Hand_Detect_Gesture():

    def __init__(self, frame):
        self.prev_frame = frame
        self.frame = frame
        self.edge_frame = None
        self.hand_colored_frame = None



# construct the argument parse and parse the arguments
def main():
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
    (grabbed, frame) = camera.read()
    height = 90
    weight = 120
    frame = cv2.flip(frame, 1)
    frame = frame[:300, :300, :]
    frame = cv2.resize(frame, (height, weight))

    numbers_hand = Hand_Detect_Number(frame)
    data = np.loadtxt("data.data", dtype=np.uint8)
    cls = data[:, 0]
    data = data[:, 1:]
    # numbers_hand.train(data, cls)
    numbers_hand.set_background(frame)

    while True:

        # last place x, y , w, h
        # grab the current frame
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        frame = frame[:300, :300, :]
        frame = cv2.resize(frame, (height, weight))
        numbers_hand.run(frame)
        numbers_hand.show()
        # print numbers_hand.predict()


        key = cv2.waitKey(10) & 0xFF  # ?

        # if the 'q' key is pressed, stop the loop
        if key == ord('s'):
            numbers_hand.set_background(frame)
        if key == ord("q"):
            break  # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
