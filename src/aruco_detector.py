import cv2
import numpy as np
import cv2.aruco as aruco

from camera import CentralCamera
from coppelia_utils import CoppeliaSimAPI

# Global variables for slider values
top_left_x = 200
top_left_y = 200
size = 111

# ArUco dictionary and parameters
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
PARAMETERS = aruco.DetectorParameters()

def get_target_corners(top_left=(200, 200), size=111):
    """
    Get the corners of the target image.
    :param top_left: Top-left corner of the target (u, v).
    :param size: Size of the target (width and height).
    :return: Corners of the target ArUco image.
    """
    return np.array([
        top_left,
        [top_left[0] + size, top_left[1]],
        [top_left[0] + size, top_left[1] + size],
        [top_left[0], top_left[1] + size]
    ])

def estimate_pose(image, marker_length, camera_matrix, dist_coeffs):
    """
    Estimate the relative pose of an ArUco tag with respect to the camera.

    Parameters:
    - image: Input image containing the ArUco tag.
    - marker_length: Physical length of the ArUco tag (in meters).
    - camera_matrix: Camera intrinsic matrix (3x3).
    - dist_coeffs: Camera distortion coefficients (1x5 or 5x1).

    Returns:
    - rvec: Rotation vector of the marker relative to the camera.
    - tvec: Translation vector of the marker relative to the camera.
    - detected: Boolean indicating whether the marker was detected.
    """
    # Define the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # Define the detector parameters
    parameters = aruco.DetectorParameters()

    # Detect the marker
    corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if ids is not None:
        # Define the 3D points of the marker in its local coordinate system
        # The marker is assumed to lie on the XY plane (Z=0)
        obj_points = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

        # Solve the PnP problem to estimate the pose
        ret, rvec, tvec = cv2.solvePnP(obj_points, corners[0], camera_matrix, dist_coeffs)

        if ret:
            return rvec, tvec, True

    return None, None, False

def draw_pose(image, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Draw the pose of the ArUco tag on the image.

    Parameters:
    - image: Input image.
    - rvec: Rotation vector of the marker.
    - tvec: Translation vector of the marker.
    - camera_matrix: Camera intrinsic matrix (3x3).
    - dist_coeffs: Camera distortion coefficients (1x5 or 5x1).
    """
    # Draw the coordinate axes on the marker
    axis_length = 0.05  # Length of the axes (in meters)
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

def detect_aruco(image, camera_matrix, dist_coeffs, marker_length=0.1):
    """
    Detect ArUco markers in an image and highlight them.
    :param image: Input image as a numpy array.
    :return: Image with detected ArUco markers highlighted.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMETERS)
    central_point = np.zeros(2)
    dP = np.zeros_like(corners)
    Z = 0

    # If markers are detected, draw them on the image
    if ids is not None:
        # Estimate the pose of the ArUco tag
        rvec, tvec, _ = estimate_pose(image, marker_length, camera_matrix, dist_coeffs)

        Z = tvec[2]

        aruco.drawDetectedMarkers(image, corners, ids)
        draw_pose(image, rvec, tvec, camera_matrix, dist_coeffs)

        # List of points (u, v) representing the polygon
        points = get_target_corners((top_left_x, top_left_y), size)
        corners = np.squeeze(corners)

        central_point = np.mean(corners, axis=0)
        central_target = np.mean(points, axis=0)

        l = 0.01
        dP = l * (central_target - central_point)

        # Reshape the points to the required format (N, 1, 2)
        points = points.reshape((-1, 1, 2))
        # Draw the polygon on the image
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)       
    
    return image, (central_point, dP), Z, ids is not None

def on_trackbar_x(val):
    """
    Callback function for the X slider.
    """
    global top_left_x
    top_left_x = val

def on_trackbar_y(val):
    """
    Callback function for the Y slider.
    """
    global top_left_y
    top_left_y = val

def main():
    # Initialize CoppeliaSim API
    coppelia = CoppeliaSimAPI()

    # Start simulation
    coppelia.start_simulation()

    f_rho = 512 / (2 * np.tan(np.pi / 6))

    cam = CentralCamera(f=f_rho, pp=(256, 256), res=(512, 512))

    # Create a window for the sliders
    cv2.namedWindow('ArUco Detection')

    # Create trackbars for adjusting the target position
    cv2.createTrackbar('Top Left X', 'ArUco Detection', top_left_x, 512, on_trackbar_x)
    cv2.createTrackbar('Top Left Y', 'ArUco Detection', top_left_y, 512, on_trackbar_y)

    try:
        coppelia.set_vision_sensor_handle('/visionSensor')

        while True:
            # Capture image from the vision sensor
            image = coppelia.get_image()

            # Detect ArUco markers
            image, (P, dP), Z, detect = detect_aruco(image, cam.K, np.zeros((5, 1)))

            if detect:
                J = cam.image_jacobian(P, Z)
                
                print(f"P: {P}")
                print(f"dP: {dP}")
                print(f"J: {J}")

                v = np.linalg.pinv(J) @ dP

                print(f"v: {v}")
            else:
                v = np.zeros(6)

            coppelia.update_camera_pose(v)

            # Display the image
            cv2.imshow('ArUco Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Step the simulation (optional, for real-time stepping)
            coppelia.step_simulation()
    finally:
        # Stop simulation
        coppelia.stop_simulation()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()