import cv2
import numpy as np

from camera import CentralCamera
from coppelia_utils import CoppeliaSimAPI

IMAGE_RESOLUTION = 512
PERSPECTIVE_ANGLE = 60

def detect_red_marker(image):
    """
    Detects the center of mass of a red circular marker in the image.
    :param image: Image as a numpy array (shape: [height, width, 3]).
    :return: Center of mass as a tuple (x, y).
    """

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the red color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the center of mass of the largest contour
    M = cv2.moments(largest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return cx, cy

def detect_blue_marker(image):
    """
    Detects the center of mass of a blue circular marker in the image.
    :param
    image: Image as a numpy array (shape: [height, width, 3]).
    :return: Center of mass as a tuple (x, y).
    """

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the blue color
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the center of mass of the largest contour
    M = cv2.moments(largest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return cx, cy


def detect_green_marker(image):
    """
    Detects the center of mass of a green circular marker in the image.
    :param image: Image as a numpy array (shape: [height, width, 3]).
    :return: Center of mass as a tuple (x, y).
    """

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the green color
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the center of mass of the largest contour
    M = cv2.moments(largest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return cx, cy

def main():
    coppelia = CoppeliaSimAPI()
    coppelia.start_simulation()

    f_rho = IMAGE_RESOLUTION / (2 * np.tan(np.deg2rad(PERSPECTIVE_ANGLE) / 2))
    
    camera = CentralCamera(
        f=f_rho, pp=(IMAGE_RESOLUTION / 2, IMAGE_RESOLUTION / 2),
        res=(IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    )

    cv2.namedWindow("Depth Image")    
    cv2.namedWindow("Camera Image")

    p_t = np.array([
        [278, 187],
        [210, 255],
        [278, 323]
    ])

    p_t = p_t - (IMAGE_RESOLUTION / 2)

    theta = np.deg2rad(45)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    p_t = (R @ (p_t.T)).T

    p_t = p_t + (IMAGE_RESOLUTION / 2)

    p_t = p_t + np.random.uniform(-100, 100, (1, 2))


    try:
        coppelia.set_vision_sensor_handle('/visionSensor')
        while True:
            
            try:
            
                image = coppelia.get_image()
                r_cx, r_cy = detect_red_marker(image)
                g_cx, g_cy = detect_green_marker(image)
                b_cx, b_cy = detect_blue_marker(image)

                cv2.circle(image, (r_cx, r_cy), 4, (0, 255, 255), -1)
                cv2.circle(image, (g_cx, g_cy), 4, (255, 0, 255), -1)
                cv2.circle(image, (b_cx, b_cy), 4, (255, 255, 0), -1)

                p = np.array([[r_cx, r_cy], [g_cx, g_cy], [b_cx, b_cy]])
                
                cv2.circle(image, tuple(p_t[0].astype(np.uint)), 4, (0, 255, 255), -1)
                cv2.circle(image, tuple(p_t[1].astype(np.uint)), 4, (255, 0, 255), -1)
                cv2.circle(image, tuple(p_t[2].astype(np.uint)), 4, (255, 255, 0), -1)
                
                depth_image = coppelia.get_depth_image()

                z = np.array([
                    depth_image[r_cy, r_cx],
                    depth_image[g_cy, g_cx],
                    depth_image[b_cy, b_cx]
                ])

                z = 10 * z  + 0.01

                print(f"z = {z}")

                J1 = camera.visjac_p(p[0], z[0])
                J2 = camera.visjac_p(p[1], z[1])
                J3 = camera.visjac_p(p[2], z[2])

                J = np.vstack((J1, J2, J3))

                damping_factor = 0.8

                # Identity matrix of size 3x3
                I = np.eye(J.shape[0])

                # Compute the damped pseudo-inverse
                J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + damping_factor**2 * I)

                _lambda = 1.0
                e = p - p_t

                print(f"e = {e.flatten().sum()}")

                v = _lambda * J_damped_pinv @ e.flatten()

                coppelia.update_camera_pose(v)
            except Exception as e:
                print(e)
                coppelia.update_camera_pose(np.zeros(6))
            
            cv2.imshow('Depth Image', depth_image)
            cv2.imshow('Camera Image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            coppelia.step_simulation()
    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
