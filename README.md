# Camera Calibration Scripts
You can get both intrinsics and extrinsics of your camera.

# 1. getting images from the video (camera)
If you already have directory with images, you can pass this step
```bash
python3 get_images.py --video (video path, camera) --image_dir (path for saved images) --sleep_time (sleep time btw taking images)
```

# 2. camera calibration (camera_calibrate.py)

Use example:
```bash
python3 camera_calibrate.py --imagess_dir (path where images are stored) --dir_path (path where to store calibration yaml) --width (board width in squares) --height (board heaight in squares) --square_size (square size (in meters)) 
```

It will do the calibration and save the calibration file which consists of intrinsics of the camera in the given directory

# 3. (Optional) camera undistort (camera_undistort.py)

You can verify the correctness of the calculated intrinsics of the camera. It shows both original and undistorted images

```bash
python3 camera_undistort.py --input_image (path to image) --yaml_file (path to calibration file)
```

# 4. Extrinsics Calculation
Calcule camera's extrinsics using chessboard as a reference frame.

```bash
python3 camera_extrinsics.py --input_image (input image path) --yaml_file (calibration file path) --width (width of the chessboard in squares) --height (height of chessboard in squares) --square_size (square size in meters)
```

You can also project 2D points to 3D using chessboard as a reference frame.