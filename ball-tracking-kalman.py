import itertools
import math
import random
from collections import deque
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import imutils
import time

from statistics import mean
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
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

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

background_buffer = []
background = None

collision_found = False
found_counter = 0
collision_location = None
collision_locations = []

counter = 0

fig = plt.figure()

# canvas = np.zeros((480,640))
# screen = pf.screen(canvas, 'Sinusoid')
c, r, w, h = 100, 100, 100, 100
# keep looping
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
x0, y0, c, r = c, r, w, h

while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
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
        # print(np.sum(edges2 / 255.0))
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

    # inja ba kalman ye pishbini anjam midim
    prediction = kalman.predict()
    x, y, w, h = 0, 0, 0, 0
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c1 = max(cnts, key=cv2.contourArea)
        ((x1, y1), radius) = cv2.minEnclosingCircle(c1)

        # only proceed if the radius meets a minimum size
        if radius > 1:
            x, y, w, h = x1, y1, radius, radius
            x0, y0, c, r = x1, y1, radius, radius
            try:
                M = cv2.moments(c1)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x1), int(y1)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), -1)
            except Exception as e:
                print(e)

    measurement = np.array([x + w / 2, y + h / 2], dtype='float64')
    cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 1,
               (255, 0, 255), 2)
    if not (x == 0 and y == 0 and w == 0 and h == 0):
        # ba kalman correct mikonim measurehasho ba panjare measurementi ke sakhtim az chize jadid
        print("kalman correct:")
        x, y, w, h = kalman.correct(measurement)
        cv2.circle(frame, (int(x), int(y)), 1,
                   (0, 255, 0), 2)
    else:
        print("kalman old predict:")
        print("x: %s, y: %s, w: %s, h: %s" % (x, y, w, h))
        # age chize khoobi nayaftimam hamoon predictione kalman ro midim behesh
        x, y, w, h = prediction
    # cv2.rectangle(frame, (int(x - c / 2), int(y - r / 2)), (int(x + c / 2), int(y + r / 2)),
    #               (0, 255, 0), 2)


    if len(pts) > 5 and all([(pts[i] is None) for i in range(1, 6)]) and not (pts[0] is None):
        c, r, w, h = x0, y0, c, r
        # keep looping
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
        x0, y0, c, r = c, r, w, h

        print("lolllll")

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

    pts.appendleft(center)

    plot = np.zeros((frame.shape[0], frame.shape[1]))
    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        # print(pts[i])
        # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        # cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        p = pts[i]
        cv2.circle(plot, p, 1, (255, 255, 255), -1)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    counter += 1

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
