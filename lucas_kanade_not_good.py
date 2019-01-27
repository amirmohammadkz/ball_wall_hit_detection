import numpy as np
import cv2 as cv

# cap = cv.VideoCapture('20181016_125931.mp4')
cap = cv.VideoCapture(0)
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.03,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# print(old_frame.shape)
# old_gray = old_frame[:, :, 2] * 4 - (old_frame[:, :, 1] + old_frame[:, :, 0])
# old_frame[:, :, 0] = old_gray
# old_frame[:, :, 1] = old_gray
# old_frame[:, :, 2] = old_gray
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# cv.imshow("tmp", old_gray)
# print(old_gray)
# print(old_gray.shape)
redLower = (170, 70, 50)
redUpper = (180, 255, 255)
hsv = cv.cvtColor(old_frame, cv.COLOR_BGR2HSV)
blurred = cv.GaussianBlur(hsv, (11, 11), 0)
mask = cv.inRange(blurred, redLower, redUpper)

p0 = cv.goodFeaturesToTrack(mask, mask=None, **feature_params)
print(p0)
# print("good features:")
# print(p0)
# Create a mask image for drawing purposesd
mask = np.zeros_like(old_frame)
i = 0
while 1:
    # print(i)
    # i = i + 1
    ret, frame = cap.read()
    # print("shape:")
    # print(frame.shape)
    # print(frame_gray.shape)
    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("original", frame_gray)
    # frame_gray = frame[:, :, 0]
    # cv.imshow("0", frame_gray)
    # frame_gray = frame[:, :, 1]
    # cv.imshow("1", frame_gray)
    frame_gray = frame[:, :, 2] * 2 - (frame[:, :, 1] + frame[:, :, 0]) / 2
    # cv.imshow("2", frame_gray)
    try:
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # print(good_new)
        # print(good_old)
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
    except Exception as e:
        pass
        # print(str(e))
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    try:
        p0 = good_new.reshape(-1, 1, 2)
    except Exception as e:
        pass

cv.destroyAllWindows()
cap.release()
