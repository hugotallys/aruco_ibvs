import cv2
import numpy as np
import cv2.aruco as aruco
import math

from camera import CentralCamera
from coppelia_utils import CoppeliaSimAPI

IMAGE_RESOLUTION = 512
PERSPECTIVE_ANGLE = 60

x = np.linspace(0, 1, IMAGE_RESOLUTION)
y = np.linspace(0, 1, IMAGE_RESOLUTION)
_X, _Y = np.meshgrid(x, y)

def draw_markers(image, corners, color):
    """
    Draw the detected ArUco markers on the image.
    :param image: Input image as a numpy array.
    :param corners: Detected corners of the ArUco markers.
    """
    for corner in corners:
        cv2.circle(image, tuple(corner.astype(int)), 4, color, -1)

    return image

def update_image():
    """
    Updates the displayed image dynamically when the sliders change.
    """
    global image, cam, coppelia
    if image is not None:
        cv2.imshow('ArUco Detection', image)

def detectRGBCircles(image):
    '''Detects a red, a green and a blue circle in image'''

    threshold = 250
    
    # Split color channels
    image_red = np.logical_and(image[:,:,2] > threshold, np.logical_and(image[:,:,0] < threshold, image[:,:,1] < threshold))
    image_green = np.logical_and(image[:,:,1] > threshold, np.logical_and(image[:,:,0] < threshold, image[:,:,2] < threshold))
    image_blue = np.logical_and(image[:,:,0] > threshold, np.logical_and(image[:,:,1] < threshold, image[:,:,2] < threshold))

    # Treatment for opencv
    image_red = 255 * image_red.astype(np.uint8)
    image_green = 255 * image_green.astype(np.uint8)
    image_blue = 255 * image_blue.astype(np.uint8)
    
    f = np.zeros(6)

    f[0] = 255 * np.sum(_X*image_red)/np.sum(image_red)
    f[1] = 255 * np.sum(_Y*image_red)/np.sum(image_red)
    f[2] = 255 * np.sum(_X*image_green)/np.sum(image_green)
    f[3] = 255 * np.sum(_Y*image_green)/np.sum(image_green)
    f[4] = 255 * np.sum(_X*image_blue)/np.sum(image_blue)
    f[5] = 255 * np.sum(_Y*image_blue)/np.sum(image_blue)

    return f

def main():
    global image, cam, coppelia

    coppelia = CoppeliaSimAPI()
    coppelia.start_simulation()

    f_rho = IMAGE_RESOLUTION / (2 * np.tan(np.deg2rad(PERSPECTIVE_ANGLE) / 2))
    cam = CentralCamera(
        f=f_rho, pp=(IMAGE_RESOLUTION / 2, IMAGE_RESOLUTION / 2),
        res=(IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    )

    # Create a window for the sliders
    cv2.namedWindow('ArUco Detection')

    try:
        coppelia.set_vision_sensor_handle('/visionSensor')

        p_target = np.array(
            [
                [232, 322],
                [300, 254],
                [232, 186]
            ]
        )

        while True:
            image = coppelia.get_image()
            
            features = detectRGBCircles(image)

            z = coppelia.get_vision_sensor_height() - 0.01

            if features.all() != 0:
                p = features.reshape(3, -1).astype(int) * 2 # ????
                
                print("RGB Features: ", p)
                print("Target Features: ", p_target)
                
                image = draw_markers(image, p, (255, 0, 255))
                image = draw_markers(image, p_target, (0, 0, 0))
            
                lambda_ = 0.8

                J1 = cam.visjac_p(p[0], z)
                J2 = cam.visjac_p(p[1, :], z)
                J3 = cam.visjac_p(p[2, :], z)
                
                J = np.vstack((J1, J2, J3))

                e = p_target[0] - p[0]

                e[1] *= -1.

                print("Error: ", e)
                
                v = lambda_ * np.linalg.pinv(J1) @ e.flatten()
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
