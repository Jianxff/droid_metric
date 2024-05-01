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
    parser.add_argument("--traj", type=str, help="trajectory file", default="./trajectory.txt")
    parser.add_argument("--poses", type=str, help="result directory", default=None)
    parser.add_argument("--viz", action='store_true', help="visualize", default=False)
    parser.add_argument("--focal", type=float, default=None, help="focal length")
    parser.add_argument("--calib", type=str, default=None, help="calib file, overwrite focal")
    parser.add_argument("--weight", type=str, default='./weights/droid.pth', help="checkpoint file")
    parser.add_argument("--global-ba-frontend", type=int, default=0, help="frequency to run global ba on frontend")
    args = parser.parse_args()

    image_dir = Path(args.rgb).resolve()
    depth_dir = None
    
    if args.depth:
        depth_dir = Path(args.depth).resolve()

    # setting droid-slam options
    opt = Droid.Options()           
    opt.weights = Path(args.weight)         # checkpoint file
    opt.disable_vis = not args.viz          # visualization
    opt.focal = args.focal                  # focal length

    # global ba on frontend, 0 (set to off) by default
    opt.global_ba_frontend = args.global_ba_frontend   # frequency to run global ba on frontend

    # camera calibration
    if args.calib:
        calib = np.loadtxt(args.calib)
        opt.intrinsic = calib[:4]
        if len(calib) > 4:
            opt.distort = calib[4:]
    elif not args.focal:
        print('no calibration or focal length provided, will use estimated values')
    
    # save trajectory
    opt.trajectory_path = Path(args.traj)
    if args.poses:
        opt.poses_dir = Path(args.poses)

    # run droid-slam
    Droid.run(image_dir, opt, depth_dir=depth_dir)
        