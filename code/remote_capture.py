import cv2 as cv

def remote_capture():
    ## Load a live image from the camera to calibrate while live
    cap = cv.VideoCapture(1) # Capture from 1 as it will be the iPhone
    images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow('Live', frame)
        key = cv.waitKey(1)
        if key == ord('c'):  # Capture image on 'c' key press
            images.append(frame)
        elif key == ord('q'):  # Quit on 'q' key press
            break
    cap.release()
    cv.destroyAllWindows()
    return images
