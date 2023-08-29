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



def undistort_image(input_image, yaml_file, output, crop = False, show = False):
    # Load coefficients
    mtx, new_mtx, dist, roi, _, _ = load_coefficients(yaml_file)
    
    original = cv2.imread(input_image)
    dst = cv2.undistort(original, mtx, dist, None, new_mtx)
    if crop:
        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]
        original = cv2.resize(original, (w, h), interpolation=1)
    stack_img = np.hstack([original, dst])
    if show:
        cv2.imshow(
            "distorted-undistorted",
            cv2.resize(stack_img, (0, 0), fx=0.5, fy=0.5, interpolation=1),
        )
        cv2.waitKey(0)
    cv2.imwrite(f"{output}.png", stack_img)
def undistort_video(yaml_file, input_video, output_video, crop=False, show = False):
    mtx, new_mtx, dist, roi, _, _ = load_coefficients(yaml_file)
    cap = cv2.VideoCapture(input_video)

    # Define the codec and create a VideoWriter object
    ret, frame = cap.read()
    h, w, _ = frame.shape
    if crop:
        _,_, w, h = roi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{output_video}.avi", fourcc, cap.get(cv2.CAP_PROP_FPS), (2*w, h))
    idx = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            dst = cv2.undistort(frame, mtx, dist, None, new_mtx)
            if crop:
                # crop the frame
                x, y, w, h = roi
                dst = dst[y : y + h, x : x + w]
                frame = cv2.resize(frame, (w, h), interpolation=1)
            stack_frame = np.hstack([frame, dst])

            # write the processed frame
            out.write(stack_frame)
            if show:
                cv2.imshow('frame', stack_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            print(f'processed {idx}/{cap.get(cv2.CAP_PROP_FRAME_COUNT)-1} frames')
            idx += 1
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

@click.command()
@click.option("--input", type=str, help="Input image or video to undistort")
@click.option("--yaml_file", type=str, help="Path to calibration file")
@click.option("--output", default = 'undistorted', type=str, help="Output name")
@click.option(
    "--crop/--no-crop", " /-c", default=False, help="Use ROI to crop undistorted image"
)
@click.option(
    "--show/--no-show", " /-s", default=False, help="Show or not"
)
def main(input, yaml_file, output, show, crop):
    if ".png" in input or ".jpg" in input:
        undistort_image(input, yaml_file, output,  crop, show)
    else:
        undistort_video(yaml_file, input, output, crop, show)
if __name__ == "__main__":
    main()
