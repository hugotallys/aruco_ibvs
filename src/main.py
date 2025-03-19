import cv2
import numpy as np

from camera import CentralCamera
from coppelia_utils import CoppeliaSimAPI

IMAGE_RESOLUTION = 512
PERSPECTIVE_ANGLE = 60

DAMPING_FACTOR = 0.005
GAIN = 0.2


def detect_markers_by_color(image):
    """
    Detects the centroids of red, green, blue, and magenta markers in the image.
    
    :param image: RGB image as a numpy array (shape: [height, width, 3]).
    :return: Dictionary with keys 'red', 'green', 'blue', 'magenta' and values as [x,y] centroids
             or None if the marker is not detected.
    """
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Red is tricky in HSV because it wraps around 0/180
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Green HSV range
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    # Blue HSV range
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    
    # Magenta HSV range
    lower_magenta = np.array([140, 100, 100])
    upper_magenta = np.array([170, 255, 255])
    
    results = {}
    
    # Detect red marker (need to handle two ranges due to HSV wrap-around)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2  # Combine the two masks
    results['red'] = find_centroid(red_mask)
    
    # Detect green marker
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    results['green'] = find_centroid(green_mask)
    
    # Detect blue marker
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    results['blue'] = find_centroid(blue_mask)
    
    # Detect magenta marker
    magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
    results['magenta'] = find_centroid(magenta_mask)
    
    return results

def find_centroid(mask):
    """
    Find the centroid of the largest blob in the mask.
    
    :param mask: Binary mask image
    :return: [x, y] centroid coordinates or None if no blob found
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate centroid only if the contour area is significant
    if cv2.contourArea(largest_contour) > 20:  # Minimum area threshold
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:  # Avoid division by zero
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return [cx, cy]
    
    return None

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
        [255, 345],
        [345, 255],
        [255, 165],
        [165, 255]
    ])

    # Pure rotation

    pure_rotation = 1.
    p_t = p_t - (IMAGE_RESOLUTION / 2)

    theta = pure_rotation * np.deg2rad(np.random.uniform(10, 30))
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    p_t = (R @ (p_t.T)).T

    p_t = p_t + (IMAGE_RESOLUTION / 2)

    # Pure translation

    pure_translation = 0.
    p_t = p_t + pure_translation * np.random.uniform(-50, 50, (1, 2))

    # Scale the square size to simulate camera zoom in/out
    # camera_zoom = 0.5
    # p_c = p_t - (IMAGE_RESOLUTION / 2)
    # p_c = camera_zoom * p_c
    # p_t = p_c + (IMAGE_RESOLUTION / 2)

    iter_count = 0
    max_iter = 200

    try:
        coppelia.set_vision_sensor_handle('/visionSensor')
        while True:
            try:
                image = coppelia.get_image()
                
                detected = detect_markers_by_color(image)

                p = np.array([
                    detected['red'],
                    detected['green'],
                    detected['blue'],
                    detected['magenta']
                ])

                cv2.circle(image, tuple(p[0].astype(np.uint)), 4, (0, 255, 0), -1)
                cv2.circle(image, tuple(p_t[0].astype(np.uint)), 4, (0, 255, 0), -1)

                cv2.circle(image, tuple(p[1].astype(np.uint)), 4, (255, 0, 0), -1)
                cv2.circle(image, tuple(p_t[1].astype(np.uint)), 4, (255, 0, 0), -1)

                cv2.circle(image, tuple(p[2].astype(np.uint)), 4, (0, 0, 255), -1)
                cv2.circle(image, tuple(p_t[2].astype(np.uint)), 4, (0, 0, 255), -1)

                cv2.circle(image, tuple(p[3].astype(np.uint)), 4, (0, 0, 0), -1)
                cv2.circle(image, tuple(p_t[3].astype(np.uint)), 4, (0, 0, 0), -1)

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

                # z = 0.5 * z

                J1_t = camera.visjac_p(p_t[0], z[0])
                J2_t = camera.visjac_p(p_t[1], z[1])
                J3_t = camera.visjac_p(p_t[2], z[2])
                J4_t = camera.visjac_p(p_t[3], z[3])

                J_t = np.vstack((J1_t, J2_t, J3_t, J4_t))

                I = np.eye(J.shape[0])

                J = 0.5 * (J + J_t)

                e = p_t - p

                J_DAMPED = J.T @ np.linalg.inv(J @ J.T + DAMPING_FACTOR ** 2 * I)

                v = GAIN * J_DAMPED @ e.flatten()

                # v = GAIN * 0.5 * (np.linalg.pinv(J) + np.linalg.pinv(J_t)) @ e.flatten()

                coppelia.update_camera_pose(v)

                if np.linalg.norm(e) < np.sqrt(4*5):
                    print("Target reached!")
                    break

                with np.printoptions(precision=2, suppress=True):
                    print(f"Error: {e}")
                    print(f"Velocity: {v}")
                    print(f"Target: {p_t}")
                    print(f"Detected: {p}")
                    print(f"Depth: {z}")
            except Exception as e:
                print(e)
                coppelia.update_camera_pose(np.zeros(6))
            
            cv2.imshow('Depth Image', depth_image)
            cv2.imshow('Camera Image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            coppelia.step_simulation()
            iter_count += 1
            if iter_count >= max_iter:
                break
    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
