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


def calculate_dif(p1, p2):
    if p1 is None or p2 is None:
        return None

    return p2[0] - p1[0], p2[1] - p1[1]


def calculate_size(p1):
    return max(calculate_distance((0, 0), p1), 0.0001)


def calculate_distance(p1, p2):
    if p1 is None or p2 is None:
        return None

    dif = calculate_dif(p1, p2)
    return max(math.sqrt(dif[0] ** 2 + dif[1] ** 2), 0.0001)


def normal_dif(p1, p2):
    if p1 is None or p2 is None:
        return None

    return calculate_dif(p1, p2)[0] / calculate_distance(p1, p2), calculate_dif(p1, p2)[1] / calculate_distance(p1, p2)


def calculate_mean(queue):
    not_now_queue = list(itertools.islice(queue, 1, len(queue)))
    not_now_queue = [i for i in not_now_queue if i is not None]
    return [sum(y) / len(y) for y in zip(*not_now_queue)]
    # return mean([i for i in not_now_queue if i is not None]) / (len([i for i in not_now_queue if i is not None]))


def best_fit_slope_and_intercept(pts):
    xs, ys = zip(*pts)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)

    return m, b


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

# keep looping
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

    if len(pts) > 1:
        # print("(p2: %s, p1: %s)" % (str(pts[0]), str(pts[1])))
        dif = normal_dif(pts[1], pts[0])
        v.appendleft(dif)

        dist.appendleft(calculate_distance(pts[1], pts[0]))
    if len(v) > 1:
        dif = calculate_dif(v[1], v[0])
        a.appendleft(dif)
    if len(a) > 1:
        dif = calculate_dif(a[1], a[0])
        a_dif.appendleft(dif)
    if len(pts) > 1 and all([(pts[i] is not None) for i in range(len(pts))]):
        print(best_fit_slope_and_intercept(pts))

    print("----")

    if len(v) >= 1 and len(a) >= 1 and len(a_dif) >= 1:
        print("(v: %s, a: %s, a_dif: %s)" % (str(v[0]), str(a[0]), str(a_dif[0])))
    a_mean = []
    a_dif_mean = []
    if len(a_dif) >= 2:
        if a_dif[0] is None and a_dif[1] is None:
            a_dif.clear()
        a_dif_mean = calculate_mean(a_dif)
        print("a_dif mean = %s" % a_dif_mean)

    if len(a) >= 2:
        if a[0] is None and a[1] is None:
            a.clear()
        a_mean = calculate_mean(a)
        print("a mean = %s" % a_mean)
    # cv2.arrowedLine(plot, pts[0], pts[0]+v[0], (0, 0, 255))

    cv2.imshow("line", plot)

    if collision_found and collision_location:
        cv2.circle(frame, collision_location, 5, (0, 255, 0))
        for collision in collision_locations:
            cv2.circle(frame, collision, 5, (0, 255, 0))

    # if not collision_found and len(dist) >= 2:
    #     print("checking distance")
    #     d1, d2 = dist[0], dist[1]
    #     if d1 and d2 and d1 > (d2 + 1.5):
    #         print("col found")
    #         collision_found = True
    #         collision_location = pts[1]
    #         cv2.circle(frame, pts[1], 20, (255, 0, 0))

    # if not collision_found and len(v) >= 2:
    #     print("checking distance")
    #     d1, d2 = v[0], v[1]
    #     if d1 and d2 and (d1[0] * d2[0]) < 0:
    #         print("col found")
    #         collision_found = True
    #         collision_location = pts[1]
    #         cv2.circle(frame, pts[1], 20, (255, 0, 0))

    # if collision_found:
    #     found_counter += 1
    #     if found_counter == 2:

    # if len(a_mean) > 0:
    #     d1, d2 = a[0], a_mean
    #     if d1 and d2:
    #         print("difference:%s" % (d1[0] - d2[0]))
    # if not collision_found and len(a_mean) > 0:
    #     d1, d2 = a[0], a_mean
    #     if d1 and d2 and (d1[0] * d2[0]) < 0 and len(a) > 3:
    #         # collision_found = True
    #         collision_location = pts[0]
    #         collision_locations.append(pts[0])
    #         cv2.circle(frame, pts[0], 20, (0, 255, 0))

    if not collision_found and len(a_dif_mean) > 0:
        d1, d2 = a_dif[0], a_dif_mean
        if d1 and d2 and len(a_dif) >= 10:
            if calculate_size(d1) / calculate_size(d2) >= 5:
                collision_found = True
                collision_location = pts[0]
                collision_locations.append(pts[0])
                cv2.circle(frame, pts[0], 20, (0, 255, 0))

    if len(a_dif_mean) > 0:
        d1, d2 = a_dif[0], a_dif_mean
        if d1 and d2 and len(a_dif) > 3:
            print("(a_dif: %s, a_dif_mean: %s, ratio: %s)" % (
                str(calculate_size(d1)), str(calculate_size(d2)), str(calculate_size(d1) / calculate_size(d2))))

    if len(pts) >= 5 and all([(pts[i] is None) for i in range(5)]):
        print("col removed")
        collision_found = False
        collision_locations = []

    if (counter + 1) % 10 == 0 and len(a_dif) > 110:
        plt.cla()
        a_dif_ds = [a_dif[i] if a_dif[i] else (None, None) for i in range(len(a_dif) - 100)]
        print(len(a_dif))
        x, y = zip(*a_dif_ds)
        t = np.array(list(range(len(a_dif) - 100)))
        plt.subplot(2, 1, 1);
        plt.cla()
        plt.plot(t, x)
        plt.subplot(2, 1, 2);
        plt.cla()
        plt.plot(t, y)
        plt.draw()
        plt.pause(0.01)

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
