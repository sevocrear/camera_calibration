import cv2
import os
from pathlib import Path
import sys
import click

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from modules.calibration_utils import save_coefficients, calibrate_chessboard


@click.command()
@click.option(
    "--images_dir", default="images", type=str, help="Input dir for calibration images"
)
@click.option(
    "--dir_path", default="data", type=str, help="Output dir for calibration file"
)
@click.option("--width", type=int, help="Width of chessboard in squares")
@click.option("--height", type=int, help="Height of chessboard in squares")
@click.option("--square_size", type=float, help="Size of square in meters")
@click.option(
    "--yaml_name",
    default="chessboard_calibration.yml",
    type=str,
    help="Name of calibration file",
)
def main(images_dir, dir_path, width, height, square_size, yaml_name):
    if yaml_name:
        out_calib_file = os.path.join(dir_path, yaml_name)
    else:
        out_calib_file = os.path.join(dir_path, "chessboard_calibration.yml")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    ret, mtx, dist, rvecs, tvecs, imgs_idxs, (H, W) = calibrate_chessboard(
        images_dir, square_size, width, height
    )
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W, H), 1, (W, H))

    save_coefficients(mtx, newcameramtx, roi, dist, out_calib_file)


if __name__ == "__main__":
    main()
