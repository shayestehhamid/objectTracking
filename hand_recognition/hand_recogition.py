from collections import deque
import argparse
import imutils
import cv2
import scipy
import numpy as np


class Hand_Detect():
    def __init__(self, frame):
        self.number = 0
        self.prev_frame = frame
        self.frame = frame
        self.edge_frame = None
        self.hand_colored_frame = None
        self.res = None
        self.color_value = np.array([26, 61, 118])
        self.sample_done = False
        self.background = None
        self.background_edge = None
        self.remove_background_frame = frame

    def histogram_equalize(self):
        yc = cv2.cvtColor(self.frame, cv2.COLOR_RGB2YCR_CB)
        cv2.equalizeHist(yc[0])
        self.frame = cv2.cvtColor(yc, cv2.COLOR_YCR_CB2RGB)

    def hand_color_detect(self, frame):
        from copy import copy
        frame = copy(frame)
        # 0 blau, 1 grun, 2 rot
        # return frame[:, :, 2]
        res1 = scipy.less(frame[:, :, 1], frame[:, :, 2])
        res2 = scipy.less(frame[:, :, 0], frame[:, :, 1])
        self.hand_colored_frame = (255 * scipy.logical_and(res1, res2)).astype(np.uint8)
        self.hand_colored_frame = cv2.erode(self.hand_colored_frame, None, iterations=2)
        self.hand_colored_frame = cv2.dilate(self.hand_colored_frame, None, iterations=3)
        return self.hand_colored_frame

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
        self.edge_frame = cv2.dilate(canny, None, iterations=2)
        return self.edge_frame

    def frame_edge(self):
        dist_edge = self.canny(self.frame) - self.background_edge
        dist_edge = cv2.erode(dist_edge, None, iterations=2)
        return dist_edge

    def set_background(self, frame):
        self.background = frame
        self.background_edge = self.canny(frame)

    def remove_background(self):
        self.remove_background_frame = np.subtract(self.frame, self.background)
        vabs = np.vectorize(abs)
        res = vabs(self.remove_background_frame)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

        # res = 255 * res
        res.astype(np.uint8)
        res = res / 100
        res = 100 * res

        self.remove_background_frame = cv2.erode(res, None, iterations=1)
        return self.remove_background_frame


class Hand_Detect_Number(Hand_Detect):
    def __init__(self, frame):
        self.number = 0
        self.prev_frame = frame
        self.frame = frame
        self.edge_frame = None
        self.hand_colored_frame = None
        self.res = None
        self.color_value = np.array([26, 61, 118])
        self.sample_done = False
        from sklearn.lda import LDA
        self.clf = LDA()
        self.background = None
        self.remove_background_frame = frame
        self.centroid = (0, 0)
        self.base_line_center = (0, 0)
        self.orientation = frame

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
        from copy import copy
        self.orientation = copy(self.frame)
        self.histogram_equalize()
        self.dist(self.frame, self.prev_frame)
        self.edge_frame = self.canny(frame)
        self.hand_colored_frame = self.hand_color_detect(frame)
        self.remove_background()
        self.canny(self.frame)
        # self.res = scipy.logical_and(self.hand_colored_frame, self.edge_frame)
        # self.res = scipy.logical_and(self.remove_background_frame, self.res)
        self.res = scipy.logical_and(self.hand_colored_frame, self.remove_background_frame)
        self.res = scipy.logical_or(self.res, self.frame_edge())
        # self.res = self.remove_background_frame
        self.res = 255 * self.res
        self.res = self.res.astype(np.uint8)
        # self.res = cv2.erode(self.res, None, iterations=2)
        # self.res = cv2.dilate(self.res, None, iterations=3)
        # self.res = cv2.erode(self.res, None, iterations=2)
        self.orientation_line()
        self.lowest_valley_point(self.base_line_func(), self.res)

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
        cv2.imshow("color", self.hand_colored_frame)
        cv2.imshow("edge", self.edge_frame)
        cv2.imshow("removed edge", self.frame_edge())
        cv2.imshow("orientation", self.orientation)

    def centroid_calc(self, frame):
        res = np.argwhere(frame > 150)
        size = res.shape[0] if res.shape[0] else 1
        xs = int(np.sum(res[:, 0]) / float(size))
        ys = int(np.sum(res[:, 1]) / float(size))
        self.centroid = (ys, xs)
        cv2.circle(self.orientation, self.centroid, color=(0, 255, 0), radius=3, thickness=-1)

    def base_line(self, frame):
        hh = 10
        size = frame.shape[0]
        end_line = frame[size - 10:, :]
        res = np.argwhere(end_line > 150)
        size_arg = res.shape[0] if res.shape[0] else 1
        xs = int(np.sum(res[:, 0]) / float(size_arg))
        ys = int(np.sum(res[:, 1]) / float(size_arg))
        self.base_line_center = (ys, xs + size - hh)
        cv2.circle(self.orientation, self.base_line_center, color=(0, 0, 255), radius=3, thickness=-1)

    def orientation_line(self):
        self.centroid_calc(self.res)
        self.base_line(self.res)
        # cv2.line(self.orientation, self.base_line_center, self.centroid, color=(255, 255, 0), thickness=1)
        line = self.base_line_func()
        prev_point = (int(line(0)), 0)
        for x in xrange(0, 120):
            y = line(x)
            point = (int(y), int(x))
            cv2.line(self.orientation, prev_point, point, color=(255, 0, 0), thickness=1)
            prev_point = point

    def base_line_func(self):
        d = (self.centroid[1] - self.base_line_center[1])
        d = d if d else 0.005
        s = (self.centroid[0] - self.base_line_center[0])
        s = s if s else 00.5
        m1 = s / float(d)
        m1 = m1 if m1 else 0.005
        c1 = (self.centroid[0] - self.centroid[1] * m1)
        # print self.centroid, self.base_line_center
        # print d, m1, c1

        def hp(x):  # line on two centers
            return m1 * x + c1

        def helper(x):  # base line
            return (float(-1) / m1) * x + self.base_line_center[0] + (float(1) / m1) * self.base_line_center[1]

        return helper

    def lowest_valley_point(self, line, frame):

        flag = True
        for intercept in xrange(frame.shape[0]):
            if not flag:
                break
            white1 = 0
            white2 = 0
            black = 0
            for x in xrange(frame.shape[0]):
                y = int(line(x) + intercept)
                try:
                    if frame[y, x] > 100 and black == 0:
                        white1 += 1
                    if frame[y, x] < 100 and white2 == 0:
                        black += 1
                    if frame[y, x] > 100 and black:
                        white2
                    if frame[y, x] < 100 and white2 and white1:
                        if black > 10 and white1 > 10 and white2 > 10:
                            res_intercept = intercept
                            flag = False
                            break
                        else:
                            white1 = white2
                            black = 0
                            white2 = 0
                except:
                    pass

        black_mid = -1
        white1 = 0
        start_white1 = -1
        start_black = -1
        start_white2 = -1
        white2 = 0
        black = 0
        for x in xrange(frame.shape[0]):
            try:
                y = line(x) + res_intercept
                print "there is intercept"
            except:
                return

            try:
                if frame[y, x] > 100 and black == 0:
                    if white1 == 0:
                        start_white1 = x
                    white1 += 1
                if frame[y, x] < 100 and white2 == 0:
                    if black == 0:
                        start_black = x
                    black += 1
                if frame[y, x] > 100 and black:
                    if white2 == 0:
                        start_white2 = x
                    white2
                if frame[y, x] < 100 and white2 and white1:
                    if black > 10 and white1 > 10 and white2 > 10:
                        black_mid = line(start_black + black / 2) + res_intercept
                        cv2.circle(self.orientation, (start_black + black / 2, black_mid), radius=3, color=(0, 0, 255), thickness=-1)
                        return start_black + black / 2, black_mid, res_intercept
                    else:
                        white1 = white2
                        start_white1 = start_white2
                        black = 0
                        white2 = 0
            except:
                print "error part2"

    def train(self, data, cls):
        self.clf.fit(data, cls)

    def predict(self):
        return self.clf.predict(self.res.reshape((1, self.res.shape[0] * self.res.shape[1])))


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
