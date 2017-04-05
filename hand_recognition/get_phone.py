import urllib
import cv2
import numpy as np
url = 'http://192.168.1.7:8080/videofeed'

stream = urllib.urlopen(url)
bytes = ''
while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    # print a, b
    if a != -1 and b != -1:
        jpg = bytes[a:b + 2]
        bytes = bytes[b + 2:]

        frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        frame = cv2.flip(frame, -1)
        print frame.shape
        cv2.imshow('cam2', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
