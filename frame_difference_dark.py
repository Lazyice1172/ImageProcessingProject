# Import
import cv2 as cv


# Function
def filter_img(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (13, 13), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    eroded = cv.erode(thresh, None, iterations=1)
    dilated = cv.dilate(eroded, None, iterations=3)

    return dilated


# Main
cap = cv.VideoCapture("videos/production_id_4029952 (2160p).mp4")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv.absdiff(frame1, frame2)
    # cv.imshow("diff", diff)

    mask = filter_img(diff)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        (x, y, w, h) = cv.boundingRect(contour)

        if cv.contourArea(contour) < 500:
            continue

        cv.rectangle(frame1, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        # cv.putText(frame1, 'Vehicle Detected', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv.LINE_AA)

    # foregroundPart = cv.bitwise_and(frame1,frame1, mask=mask)

    cv.imshow("Frame", cv.resize(frame1, None, fx=0.4, fy=0.4))
    cv.imshow("Mask", cv.resize(mask, None, fx=0.4, fy=0.4))

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv.waitKey(50) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
