import cv2
from pathlib import Path

datadir = Path('/storage/datasets/video_synthesis/')
scenes = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak', 'sear_steak']
for scene in scenes:
    # Get all cameras of a scene
    scene_path = datadir / 'videos' / scene
    cameras = sorted(scene_path.glob('*.mp4'))

    # Extract frames for each video
    for cam_path in cameras:
        cam_name = cam_path.stem
        save_path = datadir / 'frames' / scene / cam_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Read video
        vidcap = cv2.VideoCapture(str(cam_path))
        success, image = vidcap.read()
        count = 0
        while success:
            # Save frame
            img_path = str(save_path / f"frame{count}.jpg")
            cv2.imwrite(img_path, image)
            success, image = vidcap.read()
            print(f'Read a new frame from {cam_name} in {scene}: ', success)
            count += 1
        print("Total frames:", count)