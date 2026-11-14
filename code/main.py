import cv2 as cv
from remote_capture import remote_capture
from calibration import calibrate_camera

if __name__ == "__main__":
    images = remote_capture()
    chessboard_size = (9, 6)  # Example chessboard size
    square_size = 1.0  # Example square size in your preferred unit
    # TODO: Correct square size once determine actual size
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(chessboard_size, square_size, images)
    print("Calibration successful:")
    print("\tRMS Reprojection Error:", ret)
    print("\tCamera matrix:\n", mtx)
    print("\tDistortion coefficients:\n", dist)
    print("\tRotation vectors:\n", rvecs)
    print("\tTranslation vectors:\n", tvecs)