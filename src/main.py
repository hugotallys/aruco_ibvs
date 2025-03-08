import time
import os
import cv2
import numpy as np
import pandas as pd

from coppelia_utils import CoppeliaSimAPI

DATASET_SIZE = 100

def generate_synthetic_dataset(api: CoppeliaSimAPI, output_folder="dataset"):
    np.random.seed(42)

    # Create output directories
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    csv_file = os.path.join(output_folder, "poses.csv")

    # Start simulation and acquire reference image at 0r (reference pose)
    api.start_simulation()
    api.set_vision_sensor_handle("/visionSensor")
    
    # Let everything settle
    time.sleep(1)

    ref_position, ref_orientation = api.get_camera_pose()
    I0 = api.get_image()

    # Save reference image
    cv2.imwrite(f"{images_folder}/image_00000.png", cv2.cvtColor(I0, cv2.COLOR_RGB2BGR))
    ref_pose = np.hstack([ref_position, ref_orientation])
    poses = [("image_00000.png", *ref_pose)]

    # Define Gaussian sampling parameters
    first_draw_std = np.array([0.01, 0.01, 0.01, 1, 1, 2])
    second_draw_std = first_draw_std / 100

    # Generate first DATASET_SIZE samples (coarse sampling)
    for i in range(1, DATASET_SIZE + 1):
        perturbed_pose = sample_pose_around(ref_position, ref_orientation, first_draw_std)
        api.set_camera_pose(*perturbed_pose)

        # Capture and save image + pose
        image = api.get_image()
        pose = np.hstack([perturbed_pose[0], perturbed_pose[1]])

        image_filename = f"image_{i:05d}.png"
        cv2.imwrite(f"{images_folder}/{image_filename}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        poses.append((image_filename, *pose))

    # Generate additional 10% of DATASET_SIZE samples (finer sampling)
    finer_dataset_size = int(DATASET_SIZE * 0.1)
    for i in range(DATASET_SIZE + 1, DATASET_SIZE + finer_dataset_size + 1):
        perturbed_pose = sample_pose_around(ref_position, ref_orientation, second_draw_std)
        api.set_camera_pose(*perturbed_pose)

        # Capture and save image + pose
        image = api.get_image()
        pose = np.hstack([perturbed_pose[0], perturbed_pose[1]])

        image_filename = f"image_{i:05d}.png"
        cv2.imwrite(f"{images_folder}/{image_filename}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        poses.append((image_filename, *pose))

    # Save poses to CSV file
    columns = ["image_filename", "pos_x", "pos_y", "pos_z", "alpha", "beta", "gamma"]
    df = pd.DataFrame(poses, columns=columns)
    df.to_csv(csv_file, index=False)

    api.stop_simulation()
    print("Dataset generation complete!")

def sample_pose_around(reference_position, reference_orientation, std):
    """
    Sample a new pose around the reference pose using a 6DOF Gaussian draw.
    """
    delta_translation = np.random.normal(0, std[:3])
    delta_rotation = np.random.normal(0, std[3:])

    # Apply translation perturbation
    new_position = reference_position + delta_translation

    # Apply rotation perturbation (angle-axis representation)
    new_orientation = reference_orientation + delta_rotation

    # Normalize the angle values to lie in the interval [0, 180]
    new_orientation = np.mod(new_orientation, 360)
    new_orientation[new_orientation > 180] -= 360

    return new_position, new_orientation

if __name__ == "__main__":
    
    output_folder = "data"

    os.makedirs(output_folder, exist_ok=True)

    api = CoppeliaSimAPI()
    
    api.start_simulation()

    api.set_vision_sensor_handle("/visionSensor")

    position, orientation = api.get_camera_pose()

    pose = np.hstack([position, orientation])

    first_draw_std = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.4])
    
    for _ in range(100):

        perturbed_pose = sample_pose_around(position, orientation, first_draw_std)

        api.set_camera_pose(perturbed_pose[0], perturbed_pose[1])

        time.sleep(0.5)

    api.stop_simulation()

    # generate_synthetic_dataset(api, output_folder)