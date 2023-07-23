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
from modules.utils import rotationMatrixToEulerAngles
import numpy as np
import cv2
import sys
import os
import click

def draw(img, corners, imgpts):
 corner = tuple(corners[0].ravel().astype(int))
 img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0, 255), 5)
 img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
 img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5)
 return img
        
@click.command()
@click.option('--input_image', type=str, help='Input image to undistort')
@click.option('--yaml_file', type=str, help='Path to calibration file')
@click.option('--width', type=int, help='Width of chessboard in squares')
@click.option('--height', type=int, help='Height of chessboard in squares')
@click.option('--square_size', type=float, help='Size of square in meters')
@click.option('--out_yaml_dir', type=str, default="data", help='Path to output calibration file')
@click.option('--out_yaml_file', type=str, default="chessboard_final_calibration.yml", help='Name of the output calibration file')
def main(input_image, yaml_file, width, height, square_size, out_yaml_dir, out_yaml_file):
    def select_pt(event,x,y,flags,param):
        global mouseX,mouseY
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),1,(255,0,0),-1)
            
            # # TRY INVERSE PROJECTION MAPPRING 
            pts_2d = np.array([[x, y, 1]]).reshape(3,1)
            pts_3d = hom_mat_inv @ pts_2d
            pts_3d /= pts_3d[2]
            print("IPM with homography:")
            print(f"2D point {pts_2d[0:2].ravel()} (px) on the image with respect to the chess = {pts_3d[0:2].ravel()} (m)")
    CHECKERBOARD = (width, height) # width, height
    mtx, new_mtx, dist, roi, _, _ = load_coefficients(yaml_file)
    
    img = cv2.imread(input_image)

    # Определение мировых координат для 3D точек
    board_3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    board_3d[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)* square_size

    # Create a window
    window_name = 'Drawing'
    cv2.namedWindow(window_name)

    ## termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ## processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[0], CHECKERBOARD[1]), )
    if ret:
        board_2d = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        ret, rotation_vector, translation_vector = cv2.solvePnP(board_3d, board_2d, mtx, dist)
        
        axis = np.float32([[square_size*3,0,0], [0,square_size*3,0], [0,0,-square_size*3]]).reshape(-1,3)
        # project 3D points to image plane
        imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)
        img = draw(img, board_2d,imgpts)
    else:
        print('Chessboard not found')
        # TODO: add manual selection of the object points
        return
    
    print(f'rot vector:\n{rotation_vector}')
    print(f'trans vector:\n{translation_vector}') 


    rotation_mat = np.zeros(shape=(3, 3))
    R = cv2.Rodrigues(rotation_vector, rotation_mat)[0]
    P = mtx @ np.column_stack((R, translation_vector))
    print(f'Projection matrix:\n{P}')
    
    # transformation cam -> chessboard (-> measn wrt):
    rot_mat_cam_grount = R.T # R is ground -> cam
    trans_cam_ground = -np.dot(R.T,translation_vector.reshape(3,1)) # cam -> ground
    
    # Homography Matrix
    hom_mat = np.array([P[0,0], P[0,1], P[0,3], P[1,0], P[1,1], P[1, 3], P[2, 0], P[2,1 ], P[2, 3]]).reshape(3,3)
    print("Homography matrix:\n", hom_mat)

    hom_mat_inv = np.linalg.inv(hom_mat)

    save_coefficients(mtx, new_mtx, roi, dist, os.path.join(out_yaml_dir, out_yaml_file), rot_mtx= rot_mat_cam_grount, trans_vect= trans_cam_ground)
    
    cv2.setMouseCallback(window_name,select_pt)
    while(1):
        cv2.imshow(window_name,img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    
if __name__ == "__main__":
    main()