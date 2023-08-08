import cv2

import click
import time
import os
from glob import glob

def save_img(frame, image_dir, idx):
    # Save image
    idx_file =  len(glob(os.path.join(image_dir, "*.png")))
    cv2.imwrite(os.path.join(image_dir, f"{idx_file}.png"), frame)
    idx += 1
    print(f"image {idx_file} saved")
    return idx


@click.command()
@click.option("--input", default="", help="input source")
@click.option(
    "--input_type", type=click.Choice(["video", "camera"], case_sensitive=False)
)
@click.option("--image_dir", default="images", help="dir to save images")
@click.option(
    "--sleep_time", default=0.0, help="time to sleep(s) between taking the images"
)
@click.option("--viz/--no-viz", " /-v", help="show images or not", default=True)
def main(input, input_type, image_dir, sleep_time, viz):
    os.makedirs(image_dir, exist_ok=True)

    cap = cv2.VideoCapture(input)

    idx = 0
    ret, frame = cap.read()
    get_time = time.time()
    while ret:
        # Vizualization
        if viz:
            cv2.imshow("Images", frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Sleeping
        if input_type == "video":
            cap.set(cv2.CAP_PROP_POS_MSEC, (idx * sleep_time * 1000))
            idx = save_img(frame, image_dir, idx)
        else:
            if abs(get_time - time.time()) >= sleep_time:
                idx = save_img(frame, image_dir, idx)
                get_time = time.time()

        # Read frame
        ret, frame = cap.read()


if __name__ == "__main__":
    main()
