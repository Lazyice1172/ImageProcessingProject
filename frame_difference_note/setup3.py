
# Import file
import cv2 as cv
import numpy as np


# Open the video file

cap = cv.VideoCapture("../videos/pexels_videos_2443793 (2160p).mp4")

# Cap the first frame and store in the variable
ret, frame1 = cap.read()
ret, frame2 = cap.read()


# Run the Loop until the video off
while cap.isOpened():

    # Get the difference between two frame
    diff = cv.absdiff(frame1, frame2)

    gray_img = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_img, (13, 13), 0)
    ret, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)


    # Show the original frame and differed frame
    cv.imshow("Frame", frame1)
    cv.imshow("Mask", dilated)

    # Store frame 2 into frame 1
    frame1 = frame2
    # Cap the new frame and store in frame 2
    ret, frame2 = cap.read()

    # Press "q" to quick
    if cv.waitKey(50) == ord("q"):
        break

# Release memory and destroy all window
cap.release()
cv.destroyAllWindows()
