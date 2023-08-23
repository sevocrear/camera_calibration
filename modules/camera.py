import numpy as np
import cv2

class Camera:

    K = np.zeros([3, 3])
    R = np.zeros([3, 3])
    t = np.zeros([3, 1])
    P = np.zeros([3, 4])

    def __init__(self, mtx, tvec, rvec):
        self.setK(mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
        self.setR(rvec[2], rvec[1], rvec[0])
        self.setT(tvec[0], tvec[1], tvec[2])
        self.updateP()

    def setK(self, fx, fy, px, py):
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = px
        self.K[1, 2] = py
        self.K[2, 2] = 1.0

    def setR(self, y, p, r):
        Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0],
                      [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]], dtype=float)
        Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)],
                      [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]], dtype=float)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -
                      np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]], dtype=float)
        # switch axes (x = -y, y = -z, z = x)
        Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        self.R = Rs.dot(Rx.dot(Ry.dot(Rz)))

    def setT(self, XCam, YCam, ZCam):
        X = np.array([XCam, YCam, ZCam])
        self.t = -self.R.dot(X)

    def updateP(self):
        '''
        Calculate projection matrix
        '''
        Rt = np.zeros([3, 4])
        Rt[0:3, 0:3] = self.R
        Rt[0:3, 3] = self.t
        self.P = self.K.dot(Rt)

    def img2world(self, point: list):
        """Convert point from image [x,y] to the world coordinate on the ground [X, Y, 1]."""
        M = np.zeros((4, 3))
        M[0, 0] = 1
        M[1, 1] = -1
        M[3, 2] = 1
        IPM = np.linalg.inv(self.P.dot(M))
        p3D = cv2.perspectiveTransform(np.array([point,
                                                    ], dtype=np.float32)[None, :, :], IPM)
        p3D = p3D.reshape(-1, 2)
        p = p3D[0]
        P = np.array([p[0],
                        -p[1],  # invert y-axis
                        1])
        return P