import cv2
import numpy as np

from camera import CentralCamera
from coppelia_utils import CoppeliaSimAPI

IMAGE_RESOLUTION = 512
PERSPECTIVE_ANGLE = 60

DAMPING_FACTOR = 0.01
GAIN = 0.1

# Set the random seed for reproducibility
np.random.seed(42)

def detect_marker(image, lower_bound, upper_bound):
    """
    Detects the center of mass of a circular marker in the image based on the specified color bounds.
    :param image: Image as a numpy array (shape: [height, width, 3]).
    :param lower_bound: Lower bound of the HSV color range.
    :param upper_bound: Upper bound of the HSV color range.
    :return: Center of mass as a tuple (x, y).
    """
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only the specified color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the center of mass of the largest contour
        M = cv2.moments(largest_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return cx, cy
    else:
        return None, None

def detect_markers(image):
    """
    Detects the centroids of red, green, and blue markers in the image.
    :param image: Image as a numpy array (shape: [height, width, 3]).
    :return: Array of shape (2, 3) with the centroids of R, G, and B channels (in that order row-wise).
    """
    # Define the lower and upper bounds for each color channel
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_magenta = np.array([140, 100, 100])
    upper_magenta = np.array([160, 255, 255])

    # Detect the centroids of the red, green, and blue markers
    r_cx, r_cy = detect_marker(image, lower_red, upper_red)
    g_cx, g_cy = detect_marker(image, lower_green, upper_green)
    b_cx, b_cy = detect_marker(image, lower_blue, upper_blue)
    m_cx, m_cy = detect_marker(image, lower_magenta, upper_magenta)

    # Create an array with the centroids
    centroids = np.array([
        [r_cx, r_cy],
        [g_cx, g_cy],
        [b_cx, b_cy],
        [m_cx, m_cy]
    ])

    return centroids

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
        [278, 323],
        [346, 255]
    ])

    p_t = p_t - (IMAGE_RESOLUTION / 2)

    theta = 0 # np.deg2rad(np.random.uniform(0, 360))
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
                
                p = detect_markers(image)

                cv2.circle(image, tuple(p[0].astype(np.uint)), 4, (0, 255, 255), -1)
                cv2.circle(image, tuple(p[1].astype(np.uint)), 4, (255, 0, 255), -1)
                cv2.circle(image, tuple(p[2].astype(np.uint)), 4, (255, 255, 0), -1)
                cv2.circle(image, tuple(p[3].astype(np.uint)), 4, (0, 0, 255), -1)

                cv2.circle(image, tuple(p_t[0].astype(np.uint)), 4, (0, 255, 255), -1)
                cv2.circle(image, tuple(p_t[1].astype(np.uint)), 4, (255, 0, 255), -1)
                cv2.circle(image, tuple(p_t[2].astype(np.uint)), 4, (255, 255, 0), -1)
                cv2.circle(image, tuple(p_t[3].astype(np.uint)), 4, (0, 0, 255), -1)

                depth_image = coppelia.get_depth_image()

                z = np.array([
                    depth_image[p[0][0], p[0][1]],
                    depth_image[p[1][0], p[1][1]],
                    depth_image[p[2][0], p[2][1]],
                    depth_image[p[3][0], p[3][1]]
                ])

                z = 10 * z  + 0.01

                J1 = camera.visjac_p(p[0], z[0])
                J2 = camera.visjac_p(p[1], z[1])
                J3 = camera.visjac_p(p[2], z[2])
                J4 = camera.visjac_p(p[3], z[3])

                J = np.vstack((J1, J2, J3, J4))

                # Identity matrix of size 3x3
                I = np.eye(J.shape[0])

                # Compute the damped pseudo-inverse
                J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + DAMPING_FACTOR**2 * I)

                e = p_t - p

                v = GAIN * J_damped_pinv @ e.flatten()
                
                # only 4 decimals precision print

                with np.printoptions(precision=4, suppress=True):
                    # print(f"J={J}")
                    # print(f"J_damped_pinv={J_damped_pinv}")
                    # print(f"e={e}")
                    print(f"v={v}")

                # v = np.zeros(6)

                v[0] = -v[0]
                v[3:6] = 0.

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
