import numpy as np
import math
import cv2

def euler_to_rot_mat(eulers, inverse = False) -> np.array:
    """
    Convert euler angles to rotation matrix.

    Inputs:
    * [roll, pitch, yaw] -> list

    Outputs:
    * rotation matrix -> np.array(3x3)
    """
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(eulers[0]), -np.sin(eulers[0])],
            [0, np.sin(eulers[0]), np.cos(eulers[0])],
        ]
    )

    rot_y = np.array(
        [
            [np.cos(eulers[1]), 0, np.sin(eulers[1])],
            [0, 1, 0],
            [-np.sin(eulers[1]), 0, np.cos(eulers[1])],
        ]
    )

    rot_z = np.array(
        [
            [np.cos(eulers[2]), -np.sin(eulers[2]), 0],
            [np.sin(eulers[2]), np.cos(eulers[2]), 0],
            [0, 0, 1],
        ]
    )

    if not inverse:
        R = rot_z @ rot_y @ rot_x
    else:
        R = rot_x @ rot_y @ rot_z
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    II = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(II - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class transforms:
    @staticmethod
    def rot_trans_to_mat(rot_mat, t) -> np.array:
        """
        Merge rotation matrix and translation vector
        into homogeneous transformation matrix (H).

            | R t |
        H = | 0 1 |

        where
        * R - rotation matrix;
        * t - translation vector.

        Inputs:
        * rot_mat (rotation matrix) -> np.array(3x3)
        * t (translation vector) -> np.array(3x1)

        Outputs:
        * mat (H matrix) -> np.array(4x4)
        """
        mat = np.eye(4)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = t
        return mat

    @staticmethod
    def calc_homotrans_cam_opt_wrt_ground(
        cam_world
    ) -> np.array:
        

        # Get rotation matrix camera optical frame -> camera frame
        rot_mat_cam_opt_fr_cam_fr = euler_to_rot_mat([-np.pi / 2, 0, -np.pi / 2])

        # Get transformation: camera optical frame -> ground frame
        homotrans_cam_opt_wrt_ground = (
            cam_world @ transforms.rot_trans_to_mat(rot_mat_cam_opt_fr_cam_fr, np.zeros((3,)))
        )
        return homotrans_cam_opt_wrt_ground

    @staticmethod
    def project_world_pts_onto_img(
        points, dist_coeffs, cam_mat, homotrans_ground_wrt_cam_opt
    ) -> np.array:
        """
        Project given points in the camera optical frame onto the
        image frame.

        Inputs:

        * points - N points in 3D (x, y, z) -> np.array(N x 3)
        dist_coeffs - distortion coefficients of the camera -> np.array(5x1)
        * cam_mat - intrinsics matrix of the camera -> np.array(3x3)
        * homotrans_ground_wrt_cam_opt - homogeneous transformation of the ground wrt the camera frame

        Outputs:
        * array of image points (x, y) -> np.array(Nx2)
        """
        # world point array
        p3ds = np.hstack([points, np.ones((points.shape[0], 1))])
        # point in the world wrt optical camera frame

        p3d_cam_opt_fr = (homotrans_ground_wrt_cam_opt @ p3ds.T).T

        p_img, _ = cv2.projectPoints(
            p3d_cam_opt_fr[:, :3].reshape((points.shape[0], 3)).astype(np.float32).T,
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            cam_mat,
            dist_coeffs,
        )
        p_img = p_img[:, 0]
        # Filter out rows with negative elements
        mask = np.all(p_img < np.inf, axis=1)
        p_img = p_img[mask]
        p_img = p_img[p_img[:, 1].argsort()]
        return p_img

    @staticmethod
    def invert_homotrans(homotrans_mat: np.array) -> np.array:
        """
        Get the inverse of this transformation matrix.

        Inputs:
        * homotrans_mat - homogeneous transformation matrix.
        Outputs:
        * homotrans_mat_inv - homotrans matrix.
        """
        homotrans_mat_inv = np.eye(4)
        homotrans_mat_inv[:3, :3] = homotrans_mat[:3, :3].T
        homotrans_mat_inv[:3, 3] = homotrans_mat[:3, :3].T @ homotrans_mat[:3, 3]
        return homotrans_mat_inv


class horizon_estimation:
    @staticmethod
    def get_horizon_pts(
        horizon_dist: float, angle_max: float = np.pi / 10, num_pts: int = 5
    ):
        """
        Calculates horizon points (X, Y, Z) in given FOV of the camera in the world frame.

        Inputs:
        * angle_max - maximum horizontal FOV of the camera -> float (rad)
        * horizon_dist - distance to the horizon (m) -> float
        * num_pts - number of horizon points to calculate -> int

        """
        range_angles = np.linspace(-angle_max, angle_max, num=num_pts)
        range_angles_tans = horizon_dist * np.tan(range_angles)
        horizon_points = np.hstack(
            [
                np.full((num_pts, 1), horizon_dist),
                range_angles_tans.reshape((num_pts, 1)),
                np.zeros((num_pts, 1)),
            ]
        )
        return horizon_points

    @staticmethod
    def extract_edge_pts(pts_2d, shape):
        """
        Given N points, polyfit them to get line coefficients,
        after that get edge image pts of the line.


        Inputs:
        * pts_2d (np.array): array of points 2D
        * shape (tuple): shape of the image (H, W, C)
        Outputs:
        * horizon_p_0_y, horizon_p_w_y (float): horizon x-s on the both edges of the image
        """
        pts_2d_xs = pts_2d[:, 0]
        pts_2d_ys = pts_2d[:, 1]

        # Get lane coefficients and get edge horizon points in the image
        lane_coeffs = np.polyfit(pts_2d_xs, pts_2d_ys, 1)
        horizon_p_0_y = int(np.polyval(lane_coeffs, 0))  # x == 0
        horizon_p_w_y = int(np.polyval(lane_coeffs, shape[1]))  # x == image.width

        return horizon_p_0_y, horizon_p_w_y
