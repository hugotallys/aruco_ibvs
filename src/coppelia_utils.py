import cv2
import numpy as np

from quaternion import quaternion, as_float_array
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
    
    def update_camera_pose(self, camera_velocity, dt=50e-3):
        """
        Update the pose of the camera in the scene.
        :param camera_velocity: Velocity of the camera in the scene as a numpy array (shape: [6]).
        """
        
        camera_pose = self.sim.getObjectPose(self.vision_sensor_handle)

        position = np.array(camera_pose[0:3])
        orientation = quaternion(*np.roll(np.array(camera_pose[3:7]), 1))

        v = camera_velocity[0:3]
        w = camera_velocity[3:6]

        position += v * dt
        orientation += 0.5 * orientation * quaternion(0, *w) * dt
       
        orientation = np.roll(as_float_array(orientation), -1)
        camera_pose = position.tolist() + orientation.tolist()

        self.sim.setObjectPose(self.vision_sensor_handle, camera_pose)

    def step_simulation(self):
        """Advance the simulation by one step."""
        self.sim.step()

    def get_simulation_time(self):
        """Get the current simulation time."""
        return self.sim.getSimulationTime()
    
    def get_vision_sensor_height(self):
        """Get the height of the vision sensor."""
        return self.sim.getObjectPosition(self.vision_sensor_handle)[2]