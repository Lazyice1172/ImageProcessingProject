# Import
import cv2 as cv

# Function
def filter_img(frame):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    _, thresh = cv.threshold(frame, 244, 255, cv.THRESH_BINARY)
    erode = cv.erode(thresh, kernel, iterations=2)
    dilated = cv.dilate(erode, kernel, iterations=2)

    return dilated


# Main

# MOG
# Not good for shadow detect but it good for detect the contours
# object_detector = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)

# KNM
# Better shadow detect and accuracy
object_detector = cv.createBackgroundSubtractorKNN(detectShadows=True)

cap = cv.VideoCapture("chosen_video.mp4")

while True:

    ret, frame = cap.read()
    ret, frame1 = cap.read()

    #  Object detector proccesses the video to identify objects, in this case - vehicles
    obj_detector = object_detector.apply(frame, frame1)

    mask = filter_img(obj_detector)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if cv.contourArea(contour) < 400:
            continue
        cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        cv.putText(frame, 'Vehicle Detected', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv.LINE_AA)

    # cv.drawContours(frame1, contours, -1, (0, 255, 0), 3)

    #foregroundPart = cv.bitwise_and(frame,frame, mask=mask)

    cv.imshow("Frame", cv.resize(frame, None, fx=0.4, fy=0.4))
    cv.imshow("Mask", cv.resize(mask, None, fx=0.4, fy=0.4))

    if cv.waitKey(50) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
