# Import
import cv2 as cv

"""
BOOSTING 
MIL (Multiple Instance Learning) 
KCF (Kernelized Correlation Filters) 
CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
MedianFlow
TLD (Tracking Learning Detection) 
MOSSE (Minimum Output Sum of Squared Error)

GOTURN (Generic Object Tracking Using Regression Network) : Require Machine Learning model
"""

# Function

# Main

tracker = cv.legacy.TrackerCSRT.create()
cap = cv.VideoCapture("videos/highway.mp4")
ret, frame = cap.read()
bbox = cv.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    ok, box = tracker.update(frame)

    if ok:
        (x, y, w, h) = [int(v) for v in box]
        cv.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)

    cv.imshow("Frame", frame)

    if cv.waitKey(50) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()


