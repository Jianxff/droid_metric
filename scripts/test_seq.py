# standard library
from pathlib import Path
from typing import *
import os
# thrid party
import argparse
from tqdm import tqdm
import cv2
import numpy as np
# metric 3d
from modules import Metric, Droid, RGBDFusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running Metric3D')
    parser.add_argument("--rgb", help='dir for image files', type=str, required=True)
    parser.add_argument("--depth", type=str, help="depth directory", required=True)
    parser.add_argument("--poses", type=str, help="result directory", default=None)
    parser.add_argument("--mesh", type=str, help="save mesh", default=None)
    parser.add_argument("--calib", type=str, help="calib file", required=True)
    
    parser.add_argument("--depth-scale", help='depth scale factor', type=float, default=1000.0)
    parser.add_argument("--viz", action='store_true', help="visualize", default=False)
    parser.add_argument("--overwrite", action='store_true', help="overwrite existing files", default=False)

    parser.add_argument("--metric-ckpt", type=str, default='./weights/metric_depth_vit_large_800k.pth', help='checkpoint file')
    parser.add_argument("--metric-model", type=str, default='v2-L', choices=['v2-L', 'v2-S'], help='model type')
    parser.add_argument("--droid-ckpt", type=str, default='./weights/droid.pth', help="checkpoint file")
    parser.add_argument("--global-ba-frontend", type=int, default=0, help="frequency to run global ba on frontend")
    
    args = parser.parse_args()

    # directories
    rgb_dir = Path(args.rgb).resolve()
    depth_dir = Path(args.depth).resolve()
    depth_color_dir = depth_dir / 'colormap'
    pose_dir = Path(args.poses).resolve() if args.poses else Path('./poses').resolve()
    mesh_file = Path(args.mesh).resolve() if args.mesh else Path('./mesh.ply').resolve()

    os.makedirs(depth_color_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # calibration
    calib = np.loadtxt(args.calib)
    intr = calib[:4]
    distort = calib[4:] if len(calib) > 4 else None
    fxy = (intr[0] + intr[1]) / 2

    # metric 3d ###############################################################
    metric = Metric(
        checkpoint=args.metric_ckpt,
        model_name=args.metric_model
    )

    images = list(rgb_dir.glob('*.[p|j][n|p]g'))
    for image in tqdm(images):
        if os.path.exists(str(depth_dir / f'{image.stem}.png')) and not args.overwrite:
            continue

        depth = metric(image, fxy)
        # save orignal depth
        depth_u16 = (depth * args.depth_scale).astype('uint16')
        cv2.imwrite(str(depth_dir / f'{image.stem}.png'), depth_u16)
        # save colormap
        depth_color = metric.gray_to_colormap(depth)
        cv2.imwrite(str(depth_color_dir / f'{image.stem}.png'), depth_color)


    # droid slam ###############################################################
    opt = Droid.Options()  
    # basic         
    opt.weights = Path(args.droid_ckpt)     # checkpoint file
    opt.disable_vis = not args.viz          # visualization
    opt.depth_scale = args.depth_scale      # depth scale factor
    # calibration
    opt.intrinsic = intr
    opt.distort = distort
    # global ba on frontend, 0 (set to off) by default
    opt.global_ba_frontend = args.global_ba_frontend   # frequency to run global ba on frontend
    # save trajectory
    opt.poses_dir = Path(args.poses)

    # run droid-slam
    Droid.run(rgb_dir, opt, depth_dir=depth_dir)


    # fusion ###############################################################
    mesh = RGBDFusion.pipeline(
        image_dir=args.rgb,
        depth_dir=args.depth,
        traj_dir=args.poses,
        intrinsic=intr,
        distort=distort,
        depth_scale=args.depth_scale,
        viz=False
    )
    
    RGBDFusion.simplify_mesh(
        mesh=mesh,
        voxel_size=0.05,
        save=args.mesh
    )



