import numpy as np
import cv2
import time
from quaternion import quaternion, as_float_array
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class CoppeliaSimAPI:
    def __init__(self, host='localhost', port=23000):
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require('sim')
        self.vision_sensor_handle = None

    def start_simulation(self):
        self.sim.startSimulation()

    def stop_simulation(self):
        self.sim.stopSimulation()

    def set_vision_sensor_handle(self, vision_sensor_name):
        self.vision_sensor_handle = self.sim.getObjectHandle(vision_sensor_name)

    def get_image(self):
        image, resolution = self.sim.getVisionSensorImg(self.vision_sensor_handle)
        image = self.sim.unpackUInt8Table(image)
        image = np.array(image, dtype=np.uint8).reshape([resolution[1], resolution[0], 3])
        return image

    def set_camera_pose(self, position, orientation_quat):
        pose = position.tolist() + np.roll(as_float_array(orientation_quat), -1).tolist()
        self.sim.setObjectPose(self.vision_sensor_handle, -1, pose)

    def get_camera_pose(self):
        pose = self.sim.getObjectPose(self.vision_sensor_handle, -1)
        position = np.array(pose[:3])
        orientation = quaternion(*np.roll(np.array(pose[3:]), 1))
        return position, orientation

    def get_simulation_time(self):
        return self.sim.getSimulationTime()

def generate_synthetic_dataset(api: CoppeliaSimAPI, output_folder="dataset"):
    np.random.seed(42)

    # Start simulation and acquire reference image at 0r (reference pose)
    api.start_simulation()
    api.set_vision_sensor_handle("/visionSensor")
    
    # Let everything settle
    time.sleep(1)

    ref_position, ref_orientation = api.get_camera_pose()
    I0 = api.get_image()

    # Save reference image
    cv2.imwrite(f"{output_folder}/image_00000.png", cv2.cvtColor(I0, cv2.COLOR_RGB2BGR))
    np.save(f"{output_folder}/pose_00000.npy", np.hstack([ref_position, as_float_array(ref_orientation)]))

    # Define Gaussian sampling parameters
    first_draw_std = np.array([0.01, 0.01, 0.01, np.deg2rad(10), np.deg2rad(10), np.deg2rad(20)])
    second_draw_std = first_draw_std / 100

    # Generate first 10,000 samples (coarse sampling)
    for i in range(1, 10001):
        perturbed_pose = sample_pose_around(ref_position, ref_orientation, first_draw_std)
        api.set_camera_pose(*perturbed_pose)

        # Capture and save image + pose
        image = api.get_image()
        pose = np.hstack([perturbed_pose[0], as_float_array(perturbed_pose[1])])

        cv2.imwrite(f"{output_folder}/image_{i:05d}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(f"{output_folder}/pose_{i:05d}.npy", pose)

    # Generate additional 1,000 samples (finer sampling)
    for i in range(10001, 11001):
        perturbed_pose = sample_pose_around(ref_position, ref_orientation, second_draw_std)
        api.set_camera_pose(*perturbed_pose)

        # Capture and save image + pose
        image = api.get_image()
        pose = np.hstack([perturbed_pose[0], as_float_array(perturbed_pose[1])])

        cv2.imwrite(f"{output_folder}/image_{i:05d}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(f"{output_folder}/pose_{i:05d}.npy", pose)

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

    # Apply rotation perturbation (small rotation quaternion)
    dq = quaternion(1, *(delta_rotation / 2))
    new_orientation = dq * reference_orientation

    return new_position, new_orientation


if __name__ == "__main__":
    import os

    output_folder = "synthetic_dataset"
    os.makedirs(output_folder, exist_ok=True)

    api = CoppeliaSimAPI()
    generate_synthetic_dataset(api, output_folder=output_folder)
