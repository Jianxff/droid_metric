# standard library
from pathlib import Path
from typing import *
# third party
import argparse
# droid slam
from module import Droid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True, help="directory to rgb images")
    parser.add_argument("--depth", type=str, help="depth directory", default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0, help="depth scale factor")
    parser.add_argument("--viz", action='store_true', help="visualize", default=False)
    parser.add_argument("--focal", type=float, default=None, help="focal length")
    parser.add_argument("--weight", type=str, default='./weights/droid.pth', help="checkpoint file")
    args = parser.parse_args()

    image_dir = Path(args.rgb).resolve()
    depth_dir = None
    
    if args.depth:
        depth_dir = Path(args.depth).resolve()

    # run droid-slam
    opt = Droid.Options()
    opt.weights = Path(args.weight)
    opt.disable_vis = not args.viz
    opt.focal = args.focal
    opt.depth_scale = args.depth_scale
    traj = Droid.run(image_dir, opt, depth_dir=depth_dir)

    # TODO: save trajectory
    traj_file = image_dir.parent / 'traj.txt'