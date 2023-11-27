# Import
import cv2 as cv

# Function
def filter_img(frame):
    # gets Morph Ellipse shapes -> elliptic shape 
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    _, thresh = cv.threshold(frame, 250, 255, cv.THRESH_BINARY)
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

cap = cv.VideoCapture("videos/Snowy-Area.mp4")

while True:

    ret, frame = cap.read()

    #  Object detector proccesses the video to identify objects, in this case - vehicles
    obj_detector = object_detector.apply(frame)

    mask = filter_img(obj_detector)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    frameCopy = frame.copy()

    for contour in contours:

        # 100-200: notices trees but detects motorcycles. 300-400: no trees, doesnt detect motorcycles.
        if cv.contourArea(contour) < 100:
            continue
        (x, y, w, h) = cv.boundingRect(contour)

        cv.rectangle(frameCopy, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        cv.putText(frameCopy, 'Vehicle Detected', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv.LINE_AA)

    # cv.drawContours(frame1, contours, -1, (0, 255, 0), 3)

    foregroundPart = cv.bitwise_and(frame,frame, mask=mask)

    cv.imshow("Frame", cv.resize(frameCopy, None, fx=0.4, fy=0.4))
    cv.imshow("Mask", cv.resize(foregroundPart, None, fx=0.4, fy=0.4))

    if cv.waitKey(50) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
