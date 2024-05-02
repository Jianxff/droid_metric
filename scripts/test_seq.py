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
    parser.add_argument("--data", type=str, help='root directory for test, overwirte all')
    # parser.add_argument("--rgb", type=str, help='dir for image files')
    # parser.add_argument("--depth", type=str, help="depth directory")
    # parser.add_argument("--calib", type=str, help="calib file")
    # parser.add_argument("--poses", type=str, help="result directory", default=None)
    # parser.add_argument("--mesh", type=str, help="save mesh", default=None)
    # parser.add_argument("--traj", type=str, help="save trajectory opencv", default=None)
    
    parser.add_argument("--viz", action='store_true', help="visualize", default=False)
    parser.add_argument("--overwrite", action='store_true', help="overwrite existing files", default=False)
    parser.add_argument("--fusion", action='store_true', help="run rgbd fusion", default=False)
    
    parser.add_argument("--voxel-length", type=float, help="voxel length for fusion", default=0.05)
    parser.add_argument("--global-ba-frontend", type=int, help="frequency to run global ba on frontend", default=0)

    parser.add_argument("--metric-ckpt", type=str, help='checkpoint file', default='./weights/metric_depth_vit_large_800k.pth')
    parser.add_argument("--metric-model", type=str, help='model type', default='v2-L', choices=['v2-L', 'v2-S'])
    parser.add_argument("--droid-ckpt", type=str, help="checkpoint file", default='./weights/droid.pth')

    args = parser.parse_args()

    # directories
    # rgb_dir = Path(args.rgb).resolve()
    # depth_dir = Path(args.depth).resolve()
    # pose_dir = Path(args.poses).resolve() if args.poses else Path('./poses').resolve()
    # mesh_file = Path(args.mesh).resolve() if args.mesh else Path('./mesh.ply').resolve()

    # for unified data directory
    base_dir = Path(args.data)
    rgb_dir = base_dir / 'rgb'
    depth_dir = base_dir / 'depth'
    depth_color_dir = depth_dir / 'colormap'
    pose_dir = base_dir / 'poses'
    mesh_file = base_dir / 'mesh.ply'
    traj_file = base_dir / 'traj_cv.txt'
    calib_file = base_dir / 'calib.txt'

    os.makedirs(depth_color_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # calibration
    calib = np.loadtxt(calib_file)
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
        if os.path.exists(str(depth_dir / f'{image.stem}.npy')) and not args.overwrite:
            continue

        depth = metric(image, fxy)
        # save orignal depth
        np.save(str(depth_dir / f'{image.stem}.npy'), depth)
        # save colormap
        depth_color = metric.gray_to_colormap(depth)
        cv2.imwrite(str(depth_color_dir / f'{image.stem}.png'), depth_color)


    # droid slam ###############################################################
    opt = Droid.Options()  
    # basic         
    opt.weights = Path(args.droid_ckpt)     # checkpoint file
    opt.disable_vis = not args.viz          # visualization
    # calibration
    opt.intrinsic = intr
    opt.distort = distort
    # global ba on frontend, 0 (set to off) by default
    opt.global_ba_frontend = args.global_ba_frontend   # frequency to run global ba on frontend
    # save trajectory
    opt.poses_dir = pose_dir
    opt.trajectory_path = traj_file

    # run droid-slam
    Droid.run(rgb_dir, opt, depth_dir=depth_dir)


    # fusion ###############################################################
    if args.fusion:        
        mesh = RGBDFusion.pipeline(
            image_dir=rgb_dir,
            depth_dir=depth_dir,
            traj_dir=pose_dir,
            intrinsic=intr,
            distort=distort,
            viz=False,
            voxel_length=args.voxel_length,
            mesh_save=(mesh_file.parent / 'mesh_fusion_raw.ply'),
            cv_to_gl=True
        )
        
        RGBDFusion.simplify_mesh(
            mesh=mesh,
            voxel_size=0.05,
            save=mesh_file
        )



