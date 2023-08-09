import cv2
import numpy as np
import click
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from modules.calibration_utils import load_coefficients


@click.command()
@click.option("--input_image", type=str, help="Input image to undistort")
@click.option("--yaml_file", type=str, help="Path to calibration file")
@click.option(
    "--crop/--no-crop", " /-c", default=False, help="Use ROI to crop undistorted image"
)
def main(input_image, yaml_file, crop):
    # Load coefficients
    mtx, new_mtx, dist, roi, _, _ = load_coefficients(yaml_file)

    # If you resize image:
    # save_coefficients(mtx, dist, "calibration_chessboard.yml", resize_x=640/1280, resize_y= 340/1024)
    # mtx, dist = load_coefficients("calibration_chessboard.yml")

    original = cv2.imread(input_image)
    dst = cv2.undistort(original, mtx, dist, None, new_mtx)
    if crop:
        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]
        original = cv2.resize(original, (w, h), interpolation=1)
    stack_img = np.hstack([original, dst])

    cv2.imshow(
        "distorted-undistorted",
        cv2.resize(stack_img, (0, 0), fx=0.5, fy=0.5, interpolation=1),
    )
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
