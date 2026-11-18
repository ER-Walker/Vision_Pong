from camera import Camera
from aruco import ArucoDetector
from calibration import Calibration
import argparse

def calibrate():
    cam = Camera()
    images = cam.auto_capture((9, 6))
    chessboard_size = (9, 6)  # Example chessboard size
    square_size = 1.0  # Example square size in your preferred unit
    # TODO: Correct square size once determine actual size
    Calibration.calibrate_camera(chessboard_size, square_size, images)
    print("Calibration successful. Parameters saved to file.")
    cam.view_undistorted()

def play():
    #TODO: Implement game logic using ArUco marker detection
    # Current code is just for testing features
    cam = Camera()
    aruco_detector = ArucoDetector(cam)
    while True:
        aruco_detector.detect_markers_live()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Final Project: Vision_Pong",
        epilog=("modes:\n"
        "  calibrate  Capture chessboard images and save intrinsic\n"
        "             calibration to data/calibration.npz\n"
        "  play       Launch the game using the saved calibration"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["calibrate", "play"], help="Mode to run the program in")
    args = parser.parse_args()

    if args.mode == "calibrate":
        calibrate()
    elif args.mode == "play":
        play()