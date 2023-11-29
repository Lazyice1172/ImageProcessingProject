
# Import file
import cv2 as cv


# Open the video file

cap = cv.VideoCapture("../videos/pexels_videos_2443793 (2160p).mp4")

# Cap the first frame and store in the variable
ret, frame1 = cap.read()
ret, frame2 = cap.read()


# Run the Loop until the video off
while cap.isOpened():

    # Get the difference between two frame
    diff = cv.absdiff(frame1, frame2)

    # Show the original frame and differed frame
    cv.imshow("Frame", frame1)
    cv.imshow("Mask", diff)

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
