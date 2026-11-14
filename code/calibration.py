import cv2 as cv
import numpy as np

def calibrate_camera(chessboard_size, square_size: float, images: list):
    """Calibrate the camera using the provided chessboard image with corner size defined in calling function.

    Args:
        chessboard_size (tuple): Number of inner corners per a chessboard row and column (rows, columns).
        square_size (float): Size of a square in your preferred unit (e.g., meters).
        images (list): List of images containing the chessboard pattern.

    Returns:
        ret (bool): Whether the calibration was successful.
        mtx (ndarray): Camera matrix.
        dist (ndarray): Distortion coefficients.
        rvecs (list): Rotation vectors.
        tvecs (list): Translation vectors.

    Errors:
        ValueError: If there are not enough valid chessboard images for calibration (less than 10).
    """
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCornersSB(gray, chessboard_size, flags=cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE) 
        # Use SB as it is more robust than the traditional findChessboardCorners
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(objpoints) < 10:
        raise ValueError("Not enough valid chessboard images for calibration. At least 10 are required.")
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

