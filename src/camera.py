import numpy as np
from spatialmath import SE3

class CentralCamera:
    def __init__(self, f, rho=1, pp=(256, 256), res=(512, 512)):
        """
        Constructor for the CentralCamera class.

        Parameters:
        - f: Focal distance (in meters or normalized units if rho=1).
        - rho: Photosite sensor size (in meters). Defaults to 1 (normalized units).
        - pp: Principal point coordinates (in pixels). Defaults to (0, 0).
        - res: Image resolution (width, height) in pixels. Defaults to (640, 480).
        """
        self.f = f  # Focal distance
        self.rho = rho  # Photosite sensor size
        self.pp = np.array(pp)  # Principal point (cx, cy)
        self.res = np.array(res)  # Image resolution (width, height)

        # Intrinsic matrix (K)
        self.K = np.array([
            [f / rho, 0, pp[0]],
            [0, f / rho, pp[1]],
            [0, 0, 1]
        ])

    @classmethod
    def default_camera(cls):
        """
        Create a default camera with the following parameters:
        - f = 8e-3
        - rho = 10e-6
        - pp = (500, 500)
        - res = (1000, 1000)

        Returns:
        - CentralCamera object.
        """
        return cls(f=8e-3, rho=10e-6, pp=(512, 512), res=(1024, 1024))

    def project_point(self, p, pose=None):
        """
        Project 3D world points into 2D pixel coordinates.

        Parameters:
        - p: 3D world points as a numpy array of shape (3, N), where N is the number of points.
        - pose: SE3 transformation matrix representing the camera pose relative to the world frame. If None, the camera is assumed to be at the origin.

        Returns:
        - Pixel coordinates as a numpy array of shape (2, N).
        """
        # Make sure p is an array
        if not isinstance(p, np.ndarray):
            p = np.array(p)

        p = p.reshape(3, -1)

        # Add additional 1 so p is in homogeneous coordinates
        p = np.vstack((p, np.ones(p.shape[1])))

        # Apply camera pose (if provided)
        if pose is None:
            pose = SE3()

        pi = np.hstack((np.eye(3), np.zeros((3, 1))))

        # Project points using the intrinsic matrix
        p_pixel = self.K @ pi @ np.linalg.inv(pose.A) @ p

        # Normalize the pixel coordinates
        p_pixel = p_pixel[:2] / p_pixel[2]

        return p_pixel

    def visjac_p(self, p, depth):
        """
        Compute the image Jacobian for a set of points.

        Parameters:
        - p: Pixel coordinates as a numpy array of shape (n, 2), where n is the number of points.
        - depth: Depth of the points (in meters) as a numpy array of shape (n,).

        Returns:
        - Stacked image Jacobian as a numpy array of shape (2*n, 6).
        """
        # Make sure inputs are numpy arrays
        p = np.asarray(p)
        depth = np.asarray(depth)
        
        f_rho = self.f / self.rho
        n_points = p.shape[0]
        
        # Initialize the stacked Jacobian
        J_stacked = np.zeros((2 * n_points, 6))
        
        # Process each point and build the stacked Jacobian
        for i in range(n_points):
            u, v = p[i]
            
            # Shift coordinates to be relative to principal point
            u = u - self.pp[0]
            v = v - self.pp[1]
            
            # Calculate Jacobian for this point
            Ji = np.array([
                [-f_rho / depth[i], 0, u / depth[i], u * v / f_rho, -(f_rho**2 + u**2) / f_rho, v],
                [0, -f_rho / depth[i], v / depth[i], (f_rho**2 + v**2) / f_rho, -(u * v) / f_rho, -u]
            ])
            
            # Add to the stacked Jacobian
            J_stacked[2*i:2*i+2, :] = Ji
        
        return J_stacked
