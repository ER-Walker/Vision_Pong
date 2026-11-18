import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"

class Calibration:
    @staticmethod
    def calibrate_camera(chessboard_size, square_size: float, images: list) -> None:
        """Calibrate the camera using the provided chessboard image with corner size defined in calling function
        Saves the calibration parameters to a file.

        Args:
            chessboard_size (tuple): Number of inner corners per a chessboard row and column (rows, columns).
            square_size (float): Size of a square in your preferred unit (e.g., meters).
            images (list): List of images containing the chessboard pattern.

        Returns:
            None

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
        
        ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f"RMS of the calibration: {ret}") # Root Mean Square error of the calibration. Just interesting data to see how well the calibration went
        size = gray.shape[::-1]
        Calibration.save_calibration(mtx, dist, size)

    @staticmethod
    def save_calibration(mtx: MatLike, dist: MatLike, size: tuple) -> None:
        """Save the camera calibration parameters to a file.

        Args:
            mtx (MatLike): Camera matrix.
            dist (MatLike): Distortion coefficients.
            size (tuple): Image size (width, height).
        Returns:
            None
        """
        save_path = DATA_DIR / "calibration.npz" ## Saves to data from the root of project
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the data directory exists
        np.savez(save_path, mtx=mtx, dist=dist, size=size)

    @staticmethod
    def load_calibration() -> tuple:
        """Load the camera calibration parameters from a file.

        Returns:
            tuple: (mtx, dist, size) where mtx is the camera matrix, dist is the distortion coefficients, and size is the image size (width, height).
        """
        load_path = DATA_DIR / "calibration.npz"
        if not load_path.exists():
            raise FileNotFoundError("Calibration file not found.\nPlease run the calibration process first.")
        data = np.load(load_path)
        return data['mtx'], data['dist'], tuple(data['size'])

class Undistorter:
    def __init__(self, mtx, dist, size, alpha=0.0):
        # Default alpha is 0, which crops to valid pixels only
        new_mtx, self.roi = cv.getOptimalNewCameraMatrix(mtx, dist, size, alpha, size)
        self.map1, self.map2 = cv.initUndistortRectifyMap(
            mtx, dist, None, new_mtx, size, cv.CV_16SC2)
        self.new_mtx = new_mtx

    def __call__(self, frame):
        return cv.remap(frame, self.map1, self.map2, cv.INTER_LINEAR)