import cv2
import imutils
import numpy as np

img = cv2.imread("test.png", 0)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# cv2.imshow("ss",cv2.Canny(img, 40, 30))
# cv2.waitKey()
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 0.1,
#                            param1=40, param2=30, minRadius=0, maxRadius=0)
# print(len(circles))
# circles = np.uint16(np.around(circles))
#
# for i in circles[0, :]:
#     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,
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
    if radius > 3:
        try:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(cimg, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(cimg, center, 5, (0, 0, 255), -1)
        except Exception as e:
            print(e)

cv2.imshow('detected circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
