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
    parser.add_argument("--intr", help='intrinsic txt file, contained [fx, fy, cx, cy]', type=str, required=True)
    parser.add_argument("--out", help='dir for output depth', type=str, default='')
    parser.add_argument("--dmax", help='max depth', type=float, default=300.0)
    parser.add_argument("--ckpt", type=str, default='./weights/metric_depth_vit_giant2_800k.pth', help='checkpoint file')
    parser.add_argument("--model-name", type=str, default='v2-g', choices=['v2-L', 'v2-S', 'v2-g'], help='model type')
    args = parser.parse_args()

    intr = np.loadtxt(args.intr)[:4]

    metric = Metric(
        checkpoint=args.ckpt,
        model_name=args.model_name
    )

    image_dir = Path(args.images).resolve()
    images = sorted(list(image_dir.glob('*.[p|j][n|p]g')))

    out_dir = Path(args.out).resolve()
    color_dir = out_dir / 'colormap'
    os.makedirs(color_dir, exist_ok=True)

    for image in tqdm(images):
        depth = metric(rgb_image=image, intrinsic=intr, d_max=args.dmax)

        # save orignal depth
        np.save(str(out_dir / f'{image.stem}.npy'), depth)

        # save colormap
        depth_color = metric.gray_to_colormap(depth)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(color_dir / f'{image.stem}.png'), depth_color)

    


