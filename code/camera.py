import cv2 as cv
import time
from cv2.typing import MatLike, Size
import numpy as np
from pathlib import Path
from calibration import Calibration, Undistorter

# Constants
DATA_DIR = Path(__file__).parents[1] / "data"

class Camera:
    def __init__(self, cam_index: int=0):
        self.cam_index = cam_index
        self.cap = None # Initialize the camera capture object as None
        self.undistorter = None

    def open(self) -> None:
        """Open the camera for capturing if it not open already."""
        if self.cap is None:
            self.cap = cv.VideoCapture(self.cam_index)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

    def release(self) -> None:
        """Release the camera resource if it is open."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def enable_undistortion(self, alpha: float = 0.0) -> None:
        """Enable undistortion for the camera frames using the saved calibration parameters."""
        mtx, dist, size = Calibration.load_calibration()
        self.size = tuple(int(v) for v in size)
        self.undistorter = Undistorter(mtx, dist, self.size, alpha)

    def get_current_frame(self, raw: bool = False) -> MatLike:
        """Get the current frame from the camera."""
        if self.cap is None:
            self.open() 
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        if self.undistorter is not None and not raw:
            if (frame.shape[1], frame.shape[0]) != self.size:
                raise RuntimeError("Frame size does not match the expected size for undistortion.")
            frame = self.undistorter(frame)
        return frame


    def auto_capture(self, chessboard_size: tuple) -> list[MatLike]:
        """Capture images from an automatic camera capture for calibration, ensuring the chessboard is fully visible.

        Args:
            chessboard_size (tuple): Number of inner corners per a chessboard row and column (rows, columns).

        Returns:
            images (list[MatLike]): List (matrix) of captured images.
        """
        images = []
        last_time = 0
        last_center = None
        min_shift = 10  # Minimum shift in pixels to consider the chessboard moved
        min_interval = 1.0  # Minimum interval in seconds between captures
        n_images = 15  # Number of images to capture

        while len(images) < n_images:
            frame = self.get_current_frame(raw=True)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCornersSB(gray, chessboard_size, flags=cv.CALIB_CB_NORMALIZE_IMAGE)
            display = frame.copy()
            now = time.time()

            if found:  # Capture image if the chessboard is fully visible
                cv.drawChessboardCorners(display, chessboard_size, corners, found)
                center = corners.reshape(-1, 2).mean(axis=0)
                moved = last_center is None or np.linalg.norm(center - last_center) > min_shift
                if moved and now - last_time > min_interval:
                    images.append(frame)
                    image_dir = DATA_DIR / "images"
                    image_dir.mkdir(parents=True, exist_ok=True)  # Ensure the image directory exists before saving
                    cv.imwrite(f"{image_dir}/image_{len(images)}.png", frame)
                    last_time, last_center = now, center
            cv.putText(display, f"{len(images)}/{n_images}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Live', display)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.release()
        cv.destroyAllWindows()
        return images

    def view_undistorted(self, alpha: float = 0.0) -> None:
        """View the live camera feed with undistorted frames."""
        self.enable_undistortion(alpha)
        try:
            while True:
                frame = self.get_current_frame()
                cv.imshow('Undistorted', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.release()
            cv.destroyAllWindows()