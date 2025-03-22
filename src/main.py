import cv2
import numpy as np

from camera import CentralCamera
from coppelia_utils import CoppeliaSimAPI
from feature_detector import FeatureDetector

IMAGE_RESOLUTION = 512
PERSPECTIVE_ANGLE = 60

GAIN = 0.1

def main():
    coppelia = CoppeliaSimAPI()
    coppelia.start_simulation()
    
    # Create feature detector
    detector = FeatureDetector(image_resolution=IMAGE_RESOLUTION)

    f_rho = IMAGE_RESOLUTION / (2 * np.tan(np.deg2rad(PERSPECTIVE_ANGLE) / 2))
    
    camera = CentralCamera(
        f=f_rho, pp=(IMAGE_RESOLUTION / 2, IMAGE_RESOLUTION / 2),
        res=(IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    )

    cv2.namedWindow("Depth Image")    
    cv2.namedWindow("Camera Image")

    # Initialize target feature points
    p_t = np.array([
        [255, 345],
        [345, 255],
        [255, 165],
        [165, 255]
    ])

    # Apply transformations to target features
    p_t = detector.rotate_features(p_t, np.deg2rad(45))
    p_t = detector.translate_features(p_t, np.array([50, 50]))
    p_t = detector.scale_features(p_t, 1.5)

    # Define colors for the features (BGR format)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0)]

    try:
        while True:
            try:
                image = coppelia.get_image()
                
                # Detect markers in the image
                detected = detector.detect_markers_by_color(image)
                
                # Convert detected markers to array
                p = detector.get_detected_features_as_array(detected)

                # Draw detected and target features
                image = detector.draw_features(image, p, p_t, colors)

                # Assume depth is 1. for all points (both detected and target)
                z = np.full_like(p[:, 0], 1.)

                # Calculate Jacobians
                J = camera.visjac_p(p, z)
                J_t = camera.visjac_p(p_t, z)

                J = 0.5 * (J + J_t)

                # Calculate error and control velocity
                e = p_t - p
                v = GAIN * np.linalg.pinv(J) @ e.flatten()

                # Update camera pose
                coppelia.update_camera_pose(v)

                # Print debug information
                with np.printoptions(precision=2, suppress=True):
                    print(f"Error: {e}")
                    print(f"Velocity: {v}")
                    print(f"Target: {p_t}")
                    print(f"Detected: {p}")
                    print(f"Depth: {z}")
            except Exception as e:
                print(e)
                coppelia.update_camera_pose(np.zeros(6))
            
            # Display images
            cv2.imshow('Camera Image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            coppelia.step_simulation()
    finally:
        coppelia.stop_simulation()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
