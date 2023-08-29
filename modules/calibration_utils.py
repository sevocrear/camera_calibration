import cv2
from glob import glob
import numpy as np


def load_coefficients(path, export_R_T=False):
    """Loads camera matrix and distortion coefficients."""
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("MTX").mat()
    new_camera_matrix = cv_file.getNode("MTX_NEW").mat()
    roi = cv_file.getNode("ROI").mat().flatten().astype(int)
    dist_matrix = cv_file.getNode("DIST").mat()
    rotation_matrix = translation_vector = None
    if export_R_T:
        rotation_matrix = cv_file.getNode("ROT_MTX").mat()
        translation_vector = cv_file.getNode("TRANS_VECTOR").mat()
    cv_file.release()
    return [
        camera_matrix,
        new_camera_matrix,
        dist_matrix,
        roi,
        rotation_matrix,
        translation_vector,
    ]


def save_coefficients(
    mtx,
    newcameramtx,
    roi,
    dist,
    path,
    rot_mtx=None,
    eulers=None,
    trans_vect=None,
    resize_x=1,
    resize_y=1,
):
    """Save the camera matrix and the distortion coefficients to given path/file."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    mtx[0] *= resize_x
    mtx[1] *= resize_y
    newcameramtx[0] *= resize_x
    newcameramtx[1] *= resize_y
    cv_file.write("MTX", mtx)
    cv_file.write("MTX_NEW", newcameramtx)
    cv_file.write("ROI", roi)
    cv_file.write("DIST", dist)
    cv_file.write("ROT_MTX", rot_mtx)
    cv_file.write("EULERS", eulers)
    cv_file.write("TRANS_VECTOR", trans_vect)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def calibrate_chessboard(dir_path, square_size, width, height):
    """Calibrate a camera using chessboard images."""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = list(glob(f"{dir_path}/*"))
    # Iterate through all images
    images = sorted(images)

    num_unprocessed_imgs = 0
    img_idx = 0
    imgs_idxs = []
    for fname in images:
        img = cv2.imread(str(fname))
        H, W, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            (width, height),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        # If found, add object points, image points (after refining them)
        if ret:
            print(f" Processed Frame: {fname}", end="\r")
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            imgs_idxs.append(img_idx)
        else:
            num_unprocessed_imgs += 1
        img_idx += 1
        #     drawn_frame = cv2.drawChessboardCorners(img, (width, height), corners, ret)
        #     cv2.imshow("calib", drawn_frame)
        # cv2.waitKey(10)

    # Calibrate camera
    print(f"Number of unprocessed imgs:{num_unprocessed_imgs}")
    print("Please, wait. It can take a while...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))
    return [ret, mtx, dist, rvecs, tvecs, imgs_idxs, (H, W)]
