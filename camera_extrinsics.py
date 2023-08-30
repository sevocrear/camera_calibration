import cv2
import numpy as np

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from modules.calibration_utils import load_coefficients, save_coefficients
from modules.camera import Camera
from modules.utils import (
    rotationMatrixToEulerAngles,
    transforms,
    horizon_estimation,
    euler_to_rot_mat,
)
import numpy as np
import cv2
import sys
import os
import click


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 5)
    return img


@click.command()
@click.option("--input_image", type=str, help="Input image to undistort")
@click.option("--image_w", default=1280, type=int, help="Input image width")
@click.option("--image_h", default=1080, type=int, help="Input image height")
@click.option("--yaml_file", type=str, help="Path to calibration file")
@click.option("--width", type=int, help="Width of chessboard in squares")
@click.option("--height", type=int, help="Height of chessboard in squares")
@click.option("--square_size", type=float, help="Size of square in meters")
@click.option("--undistort/--no-undistort", "-u/-n", default=True, help="Undistort image")
@click.option(
    "--out_yaml_dir", type=str, default="data", help="Path to output calibration file"
)
@click.option(
    "--out_yaml_file",
    type=str,
    default="chessboard_final_calibration.yml",
    help="Name of the output calibration file",
)
def main(
    input_image,
    image_w,
    image_h,
    yaml_file,
    width,
    height,
    square_size,
    out_yaml_dir,
    out_yaml_file,
    undistort
):
    def select_pt(event, x, y, flags, param):
        global mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

            # # TRY INVERSE PROJECTION MAPPRING
            pts_2d = np.array([[x, y, 1]]).reshape(3, 1)
            pts_3d = hom_mat_inv @ pts_2d
            pts_3d /= pts_3d[2]
            print("IPM with homography:")
            print(
                f"2D point {pts_2d[0:2].ravel()} (px) on the image with respect to the chess = {pts_3d[0:2].ravel()} (m)"
            )

            p3D = calc_xy_via_homography(
                [x, y], rot_mat_cam_ground, trans_cam_ground, mtx
            )
            print(
                f"2D point {pts_2d[0:2].ravel()} (px) on the image with respect to the world frame = {p3D.ravel()} (m)"
            )

    CHECKERBOARD = (width, height)  # width, height
    mtx, new_mtx, dist, roi, _, _ = load_coefficients(yaml_file)

    # READ IMAGE
    img = cv2.imread(input_image)
    # # img = cv2.resize(img, (image_w, image_h))
    # image_h, image_w, _ = img.shape
    # UNDISTORT:  Might perform better on undistorted image
    dist1 = dist
    if undistort:
        dst = cv2.undistort(img, mtx, dist, None, new_mtx)
        img = dst
        dist1 = np.zeros((5, 1))

    # Определение мировых координат для 3D точек
    board_3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    board_3d[0, :, :2] = (
        np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
        * square_size
    )

    # Create a window
    window_name = "Drawing"
    cv2.namedWindow(window_name)

    ## termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ## processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, (CHECKERBOARD[0], CHECKERBOARD[1])
    )
    if ret:
        board_2d = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        ret, rotation_vector, translation_vector = cv2.solvePnP(
            board_3d, board_2d, mtx, dist1
        )

        axis = np.float32(
            [[square_size * 5, 0, 0], [0, square_size * 4, 0], [0, 0, -square_size * 3]]
        ).reshape(-1, 3)
        # project 3D points to image plane
        imgpts, _ = cv2.projectPoints(
            axis, rotation_vector, translation_vector, mtx, dist1
        )
        img = draw(img, board_2d, imgpts)

        rotation_mat = np.zeros(shape=(3, 3))
        R = cv2.Rodrigues(rotation_vector, rotation_mat)[0]
        P = mtx @ np.column_stack((R, translation_vector))

        # transformation cam -> chessboard (-> means wrt):
        rot_mat_cam_ground = R.T  # R is ground -> cam
        trans_cam_ground = -np.dot(
            R.T, translation_vector.reshape(3, 1)
        )  # cam -> ground
        trans_cam_ground[2] = -trans_cam_ground[2]  # z is inverted
        eulers = rotationMatrixToEulerAngles(rot_mat_cam_ground)

        # eliminate x,y offsets
        trans_cam_ground[0] = 0
        trans_cam_ground[1] = 0
        
        # Get cam frame -> world rotation
        rot_mat_cam_ground = euler_to_rot_mat(eulers)
        turns_opt = [-np.pi / 2, 0, -np.pi / 2]
        eulers_cam = rotationMatrixToEulerAngles(
            rot_mat_cam_ground @ euler_to_rot_mat(turns_opt, inverse=True)
        )
        eulers_cam[2] = 0 # eliminate yaw-angle
        # Re-calculate camera optical frame -> world rotation
        eulers = rotationMatrixToEulerAngles(
            euler_to_rot_mat(eulers_cam) @ euler_to_rot_mat(turns_opt)
        )
        rot_mat_cam_ground = euler_to_rot_mat(eulers)
    else:
        print("Chessboard not found")
        # DEBUG ---- Adjust Some shifts (to manually make it more accurate).
        trans_cam_ground[2] = 1.351
        eulers = [-1.56079633,  0.003,      -1.57079633]
        rot_mat_cam_ground = euler_to_rot_mat(eulers)
        R = rot_mat_cam_ground.T
        P = mtx @ np.column_stack((R, -R @ trans_cam_ground))
        ## ----

    # Homography Matrix
    hom_mat = np.array(
        [
            P[0, 0],
            P[0, 1],
            P[0, 3],
            P[1, 0],
            P[1, 1],
            P[1, 3],
            P[2, 0],
            P[2, 1],
            P[2, 3],
        ]
    ).reshape(3, 3)

    hom_mat_inv = np.linalg.inv(hom_mat)

    save_coefficients(
        mtx,
        new_mtx,
        roi,
        dist,
        os.path.join(out_yaml_dir, out_yaml_file),
        rot_mtx=rot_mat_cam_ground,
        eulers=rotationMatrixToEulerAngles(rot_mat_cam_ground),
        trans_vect=trans_cam_ground,
    )

    cv2.setMouseCallback(window_name, select_pt)

    # --- DEBUG calc horizon pts
    horizon_pts = horizon_estimation.get_horizon_pts(500, num_pts=20)
    R = rot_mat_cam_ground
    trans = trans_cam_ground
    H = np.hstack([R, trans])
    H = np.vstack([H, np.array([0, 0, 0, 1])])
    H[:3, :3] = H[:3, :3].T
    H[:3, 3] = -H[:3, :3] @ H[:3, 3]
    hor_pts_img = transforms.project_world_pts_onto_img(horizon_pts, dist, mtx, H)
    hor_pts_img = hor_pts_img[hor_pts_img[:, 0].argsort()]
    cv2.polylines(img, [hor_pts_img.astype(np.int32)], 0, (255, 0, 255), 3)

    print(f"trans world -> cam: \n{trans_cam_ground}")
    print(f"eulers world -> cam: \n{rotationMatrixToEulerAngles(rot_mat_cam_ground)}")
    while 1:
        cv2.imshow(window_name, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break


def calc_xy_via_homography(pixel, rot_mat, trans, mtx):
    """
    Project 2D pixel point to 3D onto the given reference frame (ground)
    """
    u, v = pixel
    # calculate transformation of the camera optical frame wrt reference frame
    R = rot_mat[:3, :3]
    translation_vector = np.array([0, 0, trans.ravel()[2]])

    # calculate it's inverse
    t_ref_opt = np.eye(4)
    t_ref_opt[:3, :3] = R.T
    t_ref_opt[:3, 3] = -R.T @ translation_vector

    # Get projection matrix
    P = mtx @ np.column_stack((R.T, -R.T @ translation_vector))
    # Homography Matrix
    hom_mat = np.array(
        [
            P[0, 0],
            P[0, 1],
            P[0, 3],
            P[1, 0],
            P[1, 1],
            P[1, 3],
            P[2, 0],
            P[2, 1],
            P[2, 3],
        ]
    ).reshape(3, 3)
    hom_mat_inv = np.linalg.inv(hom_mat)
    # TRY INVERSE PROJECTION MAPPRING
    pts_2d = np.array([[u, v, 1]]).reshape(3, 1)
    pts_3d = hom_mat_inv @ pts_2d
    pts_3d /= pts_3d[2]
    xy = pts_3d.ravel()  # x, y
    return xy


if __name__ == "__main__":
    main()
