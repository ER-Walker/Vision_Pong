import cv2 as cv
import numpy as np
from camera import Camera

class BackgroundSubtractor:
    def __init__(self, threshold: float = 30.0, kernel: int = 5, 
                 padding: int = 25, min_area: int = 400, 
                 camera: Camera = None):
        self.threshold = threshold
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel, kernel))
        self.padding = padding
        self.min_area = min_area
        self.camera = camera

    