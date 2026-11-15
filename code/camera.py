import cv2 as cv
# TODO: Capture cameras may not be at index 1, need to detect available cameras dynamically
#Might be able to get rid of this method if auto_capture works well enough
def manual_capture() -> list:
    """Capture images from a manual camera capture for calibration.

    Returns:
        images (list): List (matrix) of captured images.
    """
    ## Load a live image from the camera to calibrate while live
    cap = cv.VideoCapture(1) # Capture from 1 as it will be external camera (iPhone)
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

def auto_capture(chessboard_size: tuple) -> list:
    """Capture images from an automatic camera capture for calibration, ensuring the chessboard is fully visible.

    Args:
        chessboard_size (tuple): Number of inner corners per a chessboard row and column (rows, columns).

    Returns:
        images (list): List (matrix) of captured images.
    """
    cap = cv.VideoCapture(1) # Capture from 1 as it will be the external camera (hopefully iPhone will work)
    images = []
    count = 0 # Counter for captured images. Need at least 10
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCornersSB(gray, chessboard_size, flags=cv.CALIB_CB_NORMALIZE_IMAGE)
        cv.drawChessboardCorners(frame, chessboard_size, corners, found)
        cv.imshow('Live', frame)
        key = cv.waitKey(1)
        if found:  # Capture image if the chessboard is fully visible
            images.append(frame)
            count += 1
            if count >= 10:  # Capture 10 images
                break
    cap.release()
    cv.destroyAllWindows()
    return images