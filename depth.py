# standard library
from pathlib import Path
from typing import *
import os
# thrid party
import argparse
from tqdm import tqdm
import numpy as np
import cv2
# metric 3d
from modules import Metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running Metric3D')
    parser.add_argument("--images", help='dir for image files', type=str, required=True)
    parser.add_argument("--focal", help='focal length', type=float, default=None)
    parser.add_argument("--calib", help='calibration file, overwrite focal', type=str, default=None)
    parser.add_argument("--out", help='dir for output depth', type=str, default='')
    parser.add_argument("--depth-scale", help='depth scale factor', type=float, default=1000.0)
    parser.add_argument("--ckpt", type=str, default='./weights/metric_depth_vit_large_800k.pth', help='checkpoint file')
    parser.add_argument("--model-name", type=str, default='v2-L', choices=['v2-L', 'v2-S'], help='model type')
    args = parser.parse_args()

    if args.calib:
        calib = np.loadtxt(args.calib)
        args.focal = (calib[0] + calib[1]) / 2

    metric = Metric(
        checkpoint=args.ckpt,
        model_name=args.model_name
    )
    d_scale = args.depth_scale

    image_dir = Path(args.images).resolve()
    images = list(image_dir.glob('*.[p|j][n|p]g'))

    out_dir = Path(args.out).resolve()
    color_dir = out_dir / 'colormap'
    os.makedirs(color_dir, exist_ok=True)

    for image in tqdm(images):
        depth = metric(image, args.focal)

        # save orignal depth
        depth_u16 = (depth * d_scale).astype('uint16')
        cv2.imwrite(str(out_dir / f'{image.stem}.png'), depth_u16)

        # save colormap
        depth_color = metric.gray_to_colormap(depth)
        cv2.imwrite(str(color_dir / f'{image.stem}.png'), depth_color)

    


