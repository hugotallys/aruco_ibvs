import cv2
import numpy as np
import cv2.aruco as aruco
import math

from camera import CentralCamera
from coppelia_utils import CoppeliaSimAPI

class Corners:
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3

# Global variables for slider values
top_left_x = 200
top_left_y = 200
size = 111
yaw_angle = 0  # Rotation angle in degrees

# ArUco dictionary and parameters
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
PARAMETERS = aruco.DetectorParameters()

def get_target_corners(top_left=(200, 200), size=111, angle=0):
    """
    Get the corners of the target image and apply rotation.
    :param top_left: Top-left corner of the target (u, v).
    :param size: Size of the target (width and height).
    :param angle: Rotation angle in degrees.
    :return: Rotated corners of the target ArUco image.
    """
    # Compute center of rectangle
    center_x = top_left[0] + size / 2
    center_y = top_left[1] + size / 2

    # Define rectangle points relative to the top-left
    points = np.array([
        [top_left[0], top_left[1]],
        [top_left[0] + size, top_left[1]],
        [top_left[0] + size, top_left[1] + size],
        [top_left[0], top_left[1] + size]
    ])

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Rotation matrix
    rotation_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)]
    ])

    # Apply rotation to each point around the center
    rotated_points = np.dot(points - [center_x, center_y], rotation_matrix.T) + [center_x, center_y]

    return rotated_points.astype(int)

def draw_target_points(image, points):
    """
    Draw the target points on the image.
    :param image: Input image as a numpy array.
    :param points: List of points (u, v) representing the polygon.
    """
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

def detect_aruco(image, l=.9):
    """
    Detect ArUco markers in an image and highlight them.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMETERS)
    dP = np.zeros_like(corners)

    if ids is not None:
        # Get rotated target corners
        points = get_target_corners((top_left_x, top_left_y), size, yaw_angle)
        dP = l * (points - corners)
        aruco.drawDetectedMarkers(image, corners, ids)
        draw_target_points(image, points)
    
    return image, (np.squeeze(corners), np.squeeze(dP)), ids is not None

def update_image():
    """
    Updates the displayed image dynamically when the sliders change.
    """
    global image, cam, coppelia
    if image is not None:
        image, _, _ = detect_aruco(image, cam.K, np.zeros((5, 1)))
        cv2.imshow('ArUco Detection', image)

def on_trackbar_x(val):
    """
    Callback function for the X slider.
    """
    global top_left_x
    top_left_x = val
    # update_image()

def on_trackbar_y(val):
    """
    Callback function for the Y slider.
    """
    global top_left_y
    top_left_y = val

def on_trackbar_yaw(val):
    """
    Callback function for the Yaw rotation slider.
    """
    global yaw_angle
    yaw_angle = val

def main():
    global image, cam, coppelia

    coppelia = CoppeliaSimAPI()
    coppelia.start_simulation()

    f_rho = 512 / (2 * np.tan(np.pi / 6))
    cam = CentralCamera(f=f_rho, pp=(256, 256), res=(512, 512))

    # Create a window for the sliders
    cv2.namedWindow('ArUco Detection')

    # Create trackbars for adjusting the target position and rotation
    cv2.createTrackbar('Top Left X', 'ArUco Detection', top_left_x, 512, on_trackbar_x)
    cv2.createTrackbar('Top Left Y', 'ArUco Detection', top_left_y, 512, on_trackbar_y)
    cv2.createTrackbar('Yaw Rotation', 'ArUco Detection', yaw_angle, 360, on_trackbar_yaw)

    try:
        coppelia.set_vision_sensor_handle('/visionSensor')

        while True:
            image = coppelia.get_image()
            image, (P, dP), detect = detect_aruco(image)

            if detect:
                z = coppelia.get_vision_sensor_height()
                
                J1 = cam.visjac_p(P[0, :], z)

                J = np.vstack([J1])

                v = np.linalg.pinv(J) @ dP[0].flatten()
            else:
                v = np.zeros(6)

            coppelia.update_camera_pose(v)

            # Display the image
            cv2.imshow('ArUco Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            coppelia.step_simulation()
    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
