import numpy as np

from src.camera import CentralCamera

class TestCamera:
    
    cam = CentralCamera.default_camera()
    P = [1, 1, 5]
    
    def test_camera_projection(self):
        p0 = self.cam.project_point(self.P)
        assert (p0 == np.array([[660], [660]])).all()

    def test_camera_projection_with_displacement(self):
        # Creates SE3 transformation matrix with a translation of 0.1 in the x-axis
        Tx = np.array([[1, 0, 0, 0.1],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        p_dx = self.cam.project_point(self.P, pose=Tx)
        assert (p_dx == np.array([[644], [660]])).all()

        p0 = self.cam.project_point(self.P)
        # also test camera sensitivity
        assert (((p_dx - p0) / 0.1) == np.array([[-160], [0]])).all()

    def test_image_jacobian(self):
        p0 = self.cam.project_point(self.P)
        J = self.cam.visjac_p(p0, depth=5)
        assert np.array_equal(J, np.array([[ -160, 0, 32, 32, -832, 160], [ 0, -160, 32, 832, -32, -160]]))
