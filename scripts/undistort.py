# standard library
from pathlib import Path
from typing import *
import os 
# thrid party
import numpy as np
import argparse
import cv2
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True, help="directory to rgb images")
    parser.add_argument("--calib", type=str, help="calib file", default=None)
    parser.add_argument("--save", type=str, help="save directory", default=None)

    args = parser.parse_args()

    image_dir = Path(args.images).resolve()
    calib_file = Path(args.calib).resolve()
    save_dir = Path(args.save).resolve()

    calib = np.loadtxt(calib_file)

    if len(calib) < 4:
        raise ValueError("No distortion parameters found in calib file.")

    os.makedirs(save_dir, exist_ok=True)

    intr = calib[:4]
    dist = calib[4:]

    K = np.eye(3)
    K[0, 0], K[1, 1] = intr[0], intr[1]
    K[0, 2], K[1, 2] = intr[2], intr[3]

    images = sorted(list(image_dir.glob('*.[p|j][n|p]g')))
    h, w = cv2.imread(str(images[0]), cv2.IMREAD_COLOR).shape[:2]

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), 5)
    
    new_intr = np.array([new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2]])
    np.savetxt(str(save_dir.parent / 'calib.txt'), new_intr)

    for image in tqdm.tqdm(images):
        img = (cv2.imread(str(image), cv2.IMREAD_COLOR)).astype(np.uint8)
        undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        undist = undist[y:y+h, x:x+w]
        cv2.imwrite(str(save_dir / f'{image.stem}.png'), undist)