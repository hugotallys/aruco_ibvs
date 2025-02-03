import numpy as np

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

    def project_points(self, p, camera_pose=None):
        """
        Project 3D world points into 2D pixel coordinates.

        Parameters:
        - p: 3D world points as a numpy array of shape (3, N), where N is the number of points.
        - camera_pose: SE3 transformation matrix (4x4) representing the camera pose relative to the world frame. If None, the camera is assumed to be at the origin.

        Returns:
        - Pixel coordinates as a numpy array of shape (2, N).
        """
        # Add aditional 1 so p is in homogenous coordinates
        p = np.vstack((p, np.ones(p.shape[1])))

        # Apply camera pose (if provided)
        if camera_pose is not None:
            p = camera_pose @ p  # Transform points to camera frame

        # Project points using the intrinsic matrix
        p_pixel = self.K @ p

        return p_pixel

    def image_jacobian(self, p, Z):
        """
        Compute the image Jacobian for a set of points.

        Parameters:
        - p: pixel coordinate relative to the principal point (pixel units).
        - Z: Depth of the points (in meters).

        Returns:
        - Image Jacobian as a numpy array of shape (2, 6).
        """

        f_rho = self.f / self.rho

        u, v = p

        u = u - self.pp[0]
        v = v - self.pp[1] 

        # print(f"u: {u}, v: {v}")

        # Z = Z[0]

        J = np.array([
            [-f_rho / Z, 0, u / Z, u * v / f_rho, -(f_rho ** 2 + u ** 2) / f_rho, v],
            [0, -f_rho / Z, v / Z, (f_rho ** 2 + v ** 2) / f_rho, -u * v / f_rho, -u]
        ])

        return J
