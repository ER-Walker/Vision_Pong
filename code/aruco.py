import cv2 as cv
from camera import Camera

class ArucoDetector:
    def __init__(self, camera: Camera):
        self.camera = camera
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.parameters = cv.aruco.DetectorParameters()

    def detect_markers_live(self) -> None:
        self.camera.enable_undistortion()
        frame = self.camera.get_current_frame()
        copy = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.ArucoDetector(self.dictionary, self.parameters).detectMarkers(gray)
        if ids is not None:
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            cv.imshow('Original Frame', copy)
            cv.imshow('Detected ArUco Markers', frame)
            cv.waitKey(1)
            print(f"Detected marker IDs: {ids.flatten()}")
            cv.destroyAllWindows()
