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

def main(
    input_images: Union[str, Path],
    output_dir: Union[str, Path],
    intrinsic: Union[str, Path],
    d_max: Optional[float] = 300.0,
    overwrite: Optional[bool] = True,
    save_colormap: Optional[bool] = False,
    checkpoint: Optional[str] = './weights/metric_depth_vit_giant2_800k.pth',
    model_name: Optional[str] = 'v2-g'
) -> None:
    # load intrinsic
    intr = np.loadtxt(intrinsic)[:4]
    # init metric 3d
    metric = Metric(checkpoint=checkpoint, model_name=model_name)
    # load images
    image_dir = Path(input_images).resolve()
    images = sorted(list(image_dir.glob('*.[p|j][n|p]g')))
    # create output dir
    out_dir = Path(output_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)
    # create colormap dir
    color_dir = out_dir / 'colormap'
    if save_colormap:
        os.makedirs(color_dir, exist_ok=True)

    for image in tqdm(images):
        if overwrite or not (out_dir / f'{image.stem}.npy').exists():
            depth = metric(rgb_image=image, intrinsic=intr, d_max=d_max)
            # save orignal depth
            np.save(str(out_dir / f'{image.stem}.npy'), depth)
            # save colormap
            if save_colormap:
                depth_color = metric.gray_to_colormap(depth)
                depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(color_dir / f'{image.stem}.png'), depth_color)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running Metric3D')
    parser.add_argument("--images", help='dir for rgb image files', type=str, required=True)
    parser.add_argument("--intr", help='intrinsic txt file, contained [fx, fy, cx, cy]', type=str, required=True)
    parser.add_argument("--out", help='dir for output depth', type=str, default='')
    parser.add_argument("--out-colormap", action='store_true', help='save colormap for depth', default=False)
    parser.add_argument("--dmax", help='max depth', type=float, default=300.0)
    parser.add_argument('--skip-existed', action='store_true', help='skip existing depth file', default=False)
    parser.add_argument("--checkpoint", type=str, default='./weights/metric_depth_vit_giant2_800k.pth', help='checkpoint file')
    parser.add_argument("--model-name", type=str, default='v2-g', choices=['v2-L', 'v2-S', 'v2-g'], help='model type')
    args = parser.parse_args()

    main(
        input_images=args.rgb,
        output_dir=args.out,
        intrinsic=args.intr,
        d_max=args.dmax,
        save_colormap=args.out_colormap,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        overwrite=not args.skip_existed
    )


