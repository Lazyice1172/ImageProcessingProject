# Import
import cv2 as cv


# Function
def filter_img(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)

    return dilated


# Main
cap = cv.VideoCapture("videos/pexels_videos_2443793 (2160p).mp4")

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

    # cv.drawContours(frame1, contours, -1, (0, 255, 0), 3)

    cv.imshow("Frame", frame1)
    cv.imshow("Mask", mask)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv.waitKey(50) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
