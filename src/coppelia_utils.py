import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class CoppeliaSimAPI:
    def __init__(self, host='localhost', port=23000):
        """
        Initialize the CoppeliaSim API client.
        :param host: Host address of the CoppeliaSim ZMQ server (default: 'localhost').
        :param port: Port of the CoppeliaSim ZMQ server (default: 23000).
        """
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require('sim')
        self.vision_sensor_handle = None

    def start_simulation(self):
        """Start the simulation."""
        self.sim.startSimulation()

    def stop_simulation(self):
        """Stop the simulation."""
        self.sim.stopSimulation()

    def set_vision_sensor_handle(self, vision_sensor_name):
        self.vision_sensor_handle = self.sim.getObjectHandle(vision_sensor_name)

    def get_image(self):
        """
        Capture an image from a vision sensor.
        :param vision_sensor_name: Name of the vision sensor in CoppeliaSim.
        :return: Image as a numpy array (shape: [height, width, 3]).
        """
        
        # Capture the image
        image, resolution = self.sim.getVisionSensorImg(self.vision_sensor_handle)
        
        image = self.sim.unpackUInt8Table(image)

        # Reshape the image
        image = np.array(image, dtype=np.uint8)
        image = image.reshape([resolution[1], resolution[0], 3])

        # Flips the image vertically
        image = cv2.flip(image, 0)

        # Flips the image horizontally
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    def get_depth_image(self):
        """
        Capture a depth image from a vision sensor.
        :param vision_sensor_name: Name of the vision sensor in CoppeliaSim.
        :return: Depth image as a numpy array (shape: [height, width]).
        """
        
        # Capture the depth image
        depth_image, resolution = self.sim.getVisionSensorDepth(self.vision_sensor_handle)

        depth_image = self.sim.unpackFloatTable(depth_image)
        
        # Reshape the depth image
        depth_image = np.array(depth_image, dtype=np.float32)
        depth_image = depth_image.reshape([resolution[1], resolution[0]])

        return depth_image
    
    @staticmethod
    def get_rotation_matrix_from_euler(yaw, pitch, roll):
        """
        Get a rotation matrix from Euler angles.
        :param roll: Roll angle in radians.
        :param pitch: Pitch angle in radians.
        :param yaw: Yaw angle in radians.
        :return: Rotation matrix as a numpy array (shape: [3, 3]).
        """
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        R = R_z @ R_y @ R_x

        return R

    def update_camera_pose(self, camera_velocity):
        """
        Update the pose of the camera in the scene.
        :param camera_velocity: Velocity of the camera in the scene as a numpy array (shape: [6]).
        """
        # camera_position = self.sim.getObjectPosition(self.vision_sensor_handle)
        # camera_orientation = self.sim.getObjectOrientation(self.vision_sensor_handle)

        # yaw, pitch, roll = self.sim.alphaBetaGammaToYawPitchRoll(*camera_orientation)

        # camera_orientation = [yaw, pitch, roll]

        # print(f"Camera position = {camera_position}")
        # print(f"Camera orientation = {camera_orientation}")

        # camera_position = np.array(camera_position).reshape(-1, 1)
        # camera_orientation = np.array(camera_orientation).reshape(-1, 1)

        # print(f"Camera position = {camera_position}")
        # print(f"Camera orientation = {camera_orientation}")

        dt = self.sim.getSimulationTimeStep()

        # cam_rot = self.get_rotation_matrix_from_euler(
        #     yaw=camera_orientation[0][0], pitch=camera_orientation[1][0], roll=camera_orientation[2][0]
        # ).T

        # print(f"Camera Matrix = {cam_rot}")

        # v = np.array(camera_velocity[0:3]).reshape(-1, 1)
        # w = np.array(camera_velocity[3:6]).reshape(-1, 1)

        # print(f"Velocity = {v}")
        # print(f"Angular Velocity = {w}")

        # # Converts linear and angular velocities to world frame
        # v = cam_rot @ v
        # w = cam_rot @ w

        # # Update the camera position

        # print(f"w = {w}")
        # print(f"reversed(w) = {w[::-1]}")

        v = np.array(camera_velocity[0:3]) 
        w = np.array(camera_velocity[3:6])

        w = w[::-1]

        alpha, beta, gamma = self.sim.yawPitchRollToAlphaBetaGamma(*w)

        w = np.array([alpha, beta, gamma])

        # Set the new camera pose
        self.sim.setObjectPosition(self.vision_sensor_handle, (v * dt).tolist(), self.vision_sensor_handle)        
        self.sim.setObjectOrientation(self.vision_sensor_handle, (w * dt).tolist(), self.vision_sensor_handle)

    def step_simulation(self):
        """Advance the simulation by one step."""
        self.sim.step()

    def get_simulation_time(self):
        """Get the current simulation time."""
        return self.sim.getSimulationTime()
    
    def get_vision_sensor_height(self):
        """Get the height of the vision sensor."""
        return self.sim.getObjectPosition(self.vision_sensor_handle)[2]
    
    def get_vision_sensor_pose(self):
        """Get the pose of the vision sensor."""
        return self.sim.getObjectPose(self.vision_sensor_handle)
