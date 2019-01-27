import random
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

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
pts = deque(maxlen=args["buffer"])

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
    n = 7
    # grayO = gray.copy()
    gray = grayO.copy()
    gray = (gray + (final_mask / 255.0) * n * gray) / (n + 1)
    gray = np.uint8(gray)

    # edges = cv2.Canny(grayO, 5, 20)
    edges2 = cv2.Canny(gray, 5, 20)

    if not (background is None):
        frame1 = np.abs(grayO - background)
        frame1 = cv2.inRange(frame1, 100, 255)
        cv2.imshow("b", edges2 & frame1)
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

    cv2.imshow("dd", imutils.resize(np.vstack([mask, mask2, final_mask]), height=700))
    cv2.imshow("dddd", edges2)
    time.sleep(0.001)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
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
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 3:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
