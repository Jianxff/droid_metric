# standard library
from pathlib import Path
from typing import *
# third party
import argparse
import numpy as np
# droid slam
from modules import Droid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True, help="directory to rgb images")
    parser.add_argument("--depth", type=str, help="depth directory", default=None)
    parser.add_argument("--save_traj", type=str, help="trajectory file", default="./trajectory.txt")
    parser.add_argument("--save_poses", type=str, help="result directory", default=None)
    parser.add_argument("--viz", action='store_true', help="visualize", default=False)
    parser.add_argument("--viz-save", type=str, help="save visualization", default=None)
    parser.add_argument("--intr", type=str, default=None, help="intrinsic file, containing [fx, fy, cx, cy]")
    parser.add_argument("--weight", type=str, default='./weights/droid.pth', help="checkpoint file")
    parser.add_argument("--global-ba-frontend", type=int, default=0, help="frequency to run global ba on frontend")
    args = parser.parse_args()

    rgb_image_dir = Path(args.rgb).resolve()
    depth_dir = None
    
    if args.depth:
        depth_dir = Path(args.depth).resolve()

    # setting droid-slam options
    opt = Droid.Options()           
    opt.weights = Path(args.weight)         # checkpoint file
    opt.disable_vis = not args.viz          # visualization
    opt.vis_save = args.viz_save            # save visualization

    # global ba on frontend, 0 (set to off) by default
    opt.global_ba_frontend = args.global_ba_frontend   # frequency to run global ba on frontend

    # camera calibration
    if args.intr:
        opt.intrinsic = np.loadtxt(args.intr)[:4]
    else:
        print('no calibration provided, will use estimated values')
    
    # save trajectory
    if args.save_traj:
        opt.trajectory_path = Path(args.save_traj)
    if args.save_poses:
        opt.poses_dir = Path(args.save_poses)

    # run droid-slam
    Droid.run(rgb_image_dir, opt, depth_dir=depth_dir)
        