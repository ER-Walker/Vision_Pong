import cv2 as cv
from camera import Camera

class ArucoDetector:
    def __init__(self, camera: Camera):
        """Initializes the ArucoDetector with a camera instance.

        Args:
            camera (Camera): The camera instance to use for capturing frames.
        
        Attributes:
            camera (Camera): The camera instance used for capturing frames.
            dictionary (cv.aruco_Dictionary): The ArUco marker dictionary.
            parameters (cv.aruco_DetectorParameters): The detector parameters.
            detector (cv.aruco_ArucoDetector): The ArUco detector instance.
        """
        self.camera = camera
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.parameters = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect_markers_live(self) -> None:
        """Detects ArUco markers in the live camera feed.

        This method captures frames from the camera, converts them to grayscale,
        detects ArUco markers, and displays the original and detected frames.
        Detected marker IDs are printed to the console.
        """
        try:
            self.camera.enable_undistortion()
            frame = self.camera.get_current_frame()
            copy = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            if ids is not None:
                cv.aruco.drawDetectedMarkers(frame, corners, ids)
                cv.imshow('Original Frame', copy)
                cv.imshow('Detected ArUco Markers', frame)
                cv.waitKey(1)
                print(f"Detected marker IDs: {ids.flatten()}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cv.destroyAllWindows()
  
