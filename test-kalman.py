import random
from collections import deque

import cv2
import imutils
import numpy as np

redLower = (165, 70, 30)
redUpper = (180, 255, 255)

redLower2 = (0, 70, 30)
redUpper2 = (5, 255, 255)

# redLower = (160, 70, 50)
# redUpper = (180, 255, 255)

frame_buffer = deque(maxlen=2)

buffer_size = 10
pts = deque(maxlen=buffer_size)
v = deque(maxlen=buffer_size)
a = deque(maxlen=buffer_size)
a_dif = deque(maxlen=buffer_size)
dist = deque(maxlen=buffer_size)

background_buffer = []
background = None

collision_found = False
found_counter = 0
collision_location = None
collision_locations = []

counter = 0


def detect_one_ball(frame,background):
    # avval siah sefidesh mikonim
    frameO = imutils.resize(frame, width=600)

    frame = frameO.copy()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    grayO = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # print(np.max(hsv[:, :, 0]), np.min(hsv[:, :, 0]))

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mask2 = cv2.inRange(hsv, redLower2, redUpper2)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    final_mask = (mask.copy() | mask2.copy())

    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # print(gray.shape)
    n = 9
    # grayO = gray.copy()
    gray = grayO.copy()
    gray = (gray + (final_mask / 255.0) * n * gray) / (n + 1)
    gray = np.uint8(gray)

    # edges = cv2.Canny(grayO, 5, 20)
    edges2 = cv2.Canny(gray, 5, 20)
    masked = edges2.copy()
    if not (background is None):
        frame1 = np.abs(grayO - background)
        frame1 = cv2.inRange(frame1, 150, 255)
        # cv2.imshow("canny and background", imutils.resize(np.vstack([edges2, frame1]), height=700))
        blurred_grayO = cv2.GaussianBlur(grayO, (11, 11), 0)
        blurred_background = cv2.GaussianBlur(background, (11, 11), 0)
        frame1 = np.abs(grayO - background)
        frame1 = cv2.inRange(frame1, 7, 245)
        edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernelClose)
        # cv2.imshow("canny and background with blur", imutils.resize(np.vstack([edges2, frame1]), height=700))
        # cv2.imshow("canny&background", edges2 & frame1)
        masked = edges2 & frame1
    else:
        print(np.sum(edges2 / 255.0))
        canny_not_found = np.sum(edges2 / 255.0) < 20
        if canny_not_found:
            print("canny_not_found")
            background_buffer.append(grayO.copy())
            if len(background_buffer) == 10:
                # temp = background_buffer[0].copy()
                # for b in background_buffer[1:]:
                #     temp += b.copy()
                #
                # print(temp.shape)
                # background = temp / len(background_buffer)*1.0
                background = random.choice(background_buffer)

    # final_sobel = (sobelx+sobely)
    # final_sobel = (np.abs(sobelx)+np.abs(sobely))/2

    # cv2.imshow("dd", imutils.resize(np.vstack([mask, mask2, final_mask]), height=700))
    # cv2.imshow("canny only", edges2)
    kernelOpen = np.ones((1, 1))
    kernelClose = np.ones((1, 1))

    maskOpen = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    # cv2.imshow("open and close", imutils.resize(np.vstack([maskOpen1, maskClose1]), height=700))
    # cv2.imshow("open1 and close1", imutils.resize(np.vstack([maskOpen, maskClose]), height=700))
    # ensure at least some circles were found

    # cv2.imshow("close and close1", imutils.resize(np.vstack([maskClose1, maskClose]), height=700))
    # cv2.imshow("after open_close", maskClose1)
    # time.sleep(0.001)
    cv2.waitKey()
    # find contours in the mask and initialize the current
    # (x, y) center of the ball

    # cnts = cv2.findContours(maskClose.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(maskClose.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # only proceed if the radius meets a minimum size
        if radius > 1:
            try:
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                return int(x),int(y),int(radius),int(radius)
            except Exception as e:
                print(e)


    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    #         cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # update the points queue

    return 0, 0, 0, 0

    # vagarna soorate avvalo bede
    return faces[0]


# ye classifier sadast ke ettelaatesh ro ba tavajoh be xml i ke az opencv bara soorat download kardim dar ekhtiar mizare
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\\ColdFire\\Dropbox\\projects\\cv\\hw9\\haarcascade_frontalface_default.xml')
# too inja webcam faal mishe 0 yani webcame asli
video_capture = cv2.VideoCapture(0)
# inja videoi ke capture shod ro mikhoone framesh ro negah midare
ret, frame = video_capture.read()
# oon frame ke gerefte shod ro midim be detect_one_face ta ettelaate windowi ke face ro gerefte ro begire
c, r, w, h = detect_one_ball(frame,background)

# avvali window bara track ro misazim
state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position

# bara cherayie estefade az in parameter ha az linke zir search shod:
# https://stackoverflow.com/a/17857618/5661543
kalman = cv2.KalmanFilter(4, 2, 0)
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state

posterior = c, r, w, h
x0, y0, c, r = c, r, w, h
while True:
    # dobare in kararo anjam midim
    ret, frame = video_capture.read()
    cv2.rectangle(frame, (x0, y0), (int(x0 + c), int(y0 + r)), 255, thickness=10)
    # inja ba kalman ye pishbini anjam midim
    prediction = kalman.predict()
    # bad dobare detect mikonim
    x, y, w, h = detect_one_ball(frame,background)
    # hala age chizi doros peyda nashod ghabliaro midim behesh
    if not (w < 0.1 * c):
        x0, y0, c, r = x, y, w, h
    # else:

    measurement = np.array([x + w / 2, y + h / 2], dtype='float64')

    if not (x == 0 and y == 0 and w == 0 and h == 0):
        # ba kalman correct mikonim measurehasho ba panjare measurementi ke sakhtim az chize jadid
        x, y, w, h = kalman.correct(measurement)
    else:
        # age chize khoobi nayaftimam hamoon predictione kalman ro midim behesh
        x, y, w, h = prediction

    # injam ke bara neshoondadane resulte
    cv2.rectangle(frame, (int(x - c / 2 + 5), int(y - r / 2 + 5)), (int(x + c / 2 - 5), int(y + r / 2 - 5)),
                  (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
