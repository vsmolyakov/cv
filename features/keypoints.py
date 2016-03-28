import cv2
import numpy as np

img =  cv2.imread('landscape.png')
img2 = cv2.imread('landscape.png')
img3 = cv2.imread('landscape.png')
num_rows, num_cols = img.shape[:2]

gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

#Harris corner detector
dst = cv2.cornerHarris(gray, 4, 5, 0.04) #sharp corners
#dst = cv2.cornerHarris(gray, 14, 5, 0.04) #soft corners

#result is dilated for marking the corners
dst = cv2.dilate(dst, None)

#threshold for an optimal value (image specific)
img2[dst > 0.1*dst.max()] = [0,0,0]

cv2.imshow('Harris Corners', img2)
cv2.waitKey()

#Shi-Tomasi corner detector
corners = cv2.goodFeaturesToTrack(gray, 7, 0.05, 25)
corners = np.float32(corners)

for item in corners:
    x, y = item[0]
    cv2.circle(img3, (x,y), 5, 255,-1)
    
cv2.imshow('Top K features', img3)
cv2.waitKey()

#SIFT feature detector and descriptor
sift = cv2.SIFT()

keypoints = sift.detect(gray2, None)
#keypoints, descriptors = sift.detectAndCompute(gray2, None)
img_sift = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT features', img_sift)
cv2.waitKey()

#SURF feature detector and descriptor
surf = cv2.SURF()

#threshold for the number of keypoints
surf.hessianThreshold = 15000
kp, ds = surf.detectAndCompute(gray2, None)
img_surf = cv2.drawKeypoints(img, kp, None, (0,255,0), 4)

cv2.imshow('SURF features', img_surf)
cv2.waitKey()

#FAST feature detector
fast = cv2.FastFeatureDetector()
keypoints = fast.detect(gray2, None)

#BRIEF feature descriptor
brief = cv2.DescriptorExtractor_create("BRIEF")
keypoints, descriptors = brief.compute(gray2, keypoints)

img_fast = cv2.drawKeypoints(img, keypoints, color=(0,255,0))

cv2.imshow('FAST keypoints with non-max suppression', img_fast)
cv2.waitKey()

#ORB feature detector and descriptor
orb = cv2.ORB()
keypoints = orb.detect(gray2, None)
keypoints, descriptors = orb.compute(gray2, keypoints)
img_orb = cv2.drawKeypoints(img, keypoints, color=(0,255,0), flags=0)

cv2.imshow('ORB keypoints', img_orb)
cv2.waitKey()



