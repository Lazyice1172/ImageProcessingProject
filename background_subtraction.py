# Import
import cv2 as cv

# Function
def filter_img(frame):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    _, thresh = cv.threshold(frame, 244, 255, cv.THRESH_BINARY)
    thresh = cv.erode(thresh, kernel, iterations=2)
    dilated = cv.dilate(thresh, kernel, iterations=2)

    return dilated


# Main

# MOG
# Not good for shadow detect but it good for detect the contours
# object_detector = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)

# KNM
# Better shadow detect and accuracy
object_detector = cv.createBackgroundSubtractorKNN(detectShadows=True)

cap = cv.VideoCapture("videos/highway.mp4")

while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)
    mask = filter_img(mask)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if cv.contourArea(contour) < 100:
            continue
        cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    # cv.drawContours(frame1, contours, -1, (0, 255, 0), 3)

    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)

    if cv.waitKey(50) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
