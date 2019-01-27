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
hsv_o = cv.cvtColor(old_frame, cv.COLOR_BGR2HSV)
blurred = cv.GaussianBlur(hsv_o, (11, 11), 0)
mask = cv.inRange(blurred, redLower, redUpper)
# cv.imshow("tmp", np.concatenate(old_frame, mask))
cv.imshow("mask", mask)
cv.imshow("old_frame", old_frame)
# print(hsv[:, :, 2].shape)
# print(np.max(mask))
# print(np.amax(hsv[:, :, 2]))
# hsv[:, :, 2] = (hsv[:, :, 2] + mask * 20) / 21
# hsv[:, :, 2] += mask*100000000
for n in range(1, 10, 2):
    hsv = hsv_o.copy()
    # print(np.mean(hsv[:, :, 2]))
    # x = (hsv[:, :, 2] + mask * n) / (n + 1)
    # f = np.stack([hsv[:, :, 0], hsv[:, :, 1], x], 2)
    # print(f.shape)
    # print(hsv.shape)
    # f = cv.cvtColor(f, cv.COLOR_HSV2BGR)
    # cv.imshow("n= " + str(n), f)
    # f = np.stack([hsv[:, :, 0], hsv[:, :, 1], x], 2)
    # print(f.shape)
    # print(np.mean(f[:, :, 2]))
    # print(np.mean(x))

    hsv[:, :, 2] = (hsv[:, :, 2] + (mask/255.0) * n * hsv[:, :, 2]) / (n + 1)
    # hsv[:, :, 2] = mask
    hsv = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow("n= " + str(n), hsv)
    cv.waitKey()

# hsv = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# np.concatenate(old_frame,mask)

# p0 = cv.goodFeaturesToTrack(mask, mask=None, **feature_params)

# cv.imshow("tmp", mask)
cv.waitKey(0)
