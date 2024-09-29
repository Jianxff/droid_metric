# standard library
from pathlib import Path
from typing import *
# third party
import argparse
import numpy as np
# droid slam
from modules import Droid

def main(
    input_images: Union[str, Path],
    input_depth: Union[str, Path],
    intrinsic: Union[str, Path],
    viz: Optional[bool] = False,
    output_traj: Optional[Union[str, Path]] = None,
    output_poses: Optional[Union[str, Path]] = None,
    output_pcd: Optional[Union[str, Path]] = None,
    checkpoint: Optional[str] = './weights/droid.pth',
    global_ba_frontend: Optional[int] = 0,
) -> None:
    input_images = Path(input_images).resolve()
    input_depth = Path(input_depth).resolve()
    
    # setting droid-slam options
    droid_options = Droid.Options()
    droid_options.intrinsic = np.loadtxt(intrinsic)[:4]
    droid_options.weights = Path(checkpoint)
    droid_options.disable_vis = not viz
    droid_options.vis_save = (output_pcd if viz else None)
    droid_options.global_ba_frontend = global_ba_frontend
    droid_options.trajectory_path = output_traj
    droid_options.poses_dir = output_poses

    # run Droid-SLAM
    Droid.run(input_images, droid_options, depth_dir=input_depth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True, help="directory to rgb images")
    parser.add_argument("--depth", type=str, required=True,help="depth directory")
    parser.add_argument("--intr", type=str, required=True, help="intrinsic file, containing [fx, fy, cx, cy]")
    parser.add_argument("--viz", action='store_true', help="visualize", default=False)
    parser.add_argument("--out-traj", type=str, help="trajectory file", default="./trajectory.txt")
    parser.add_argument("--out-poses", type=str, help="result directory", default=None)
    parser.add_argument("--out-pcd", type=str, help="save visualization", default=None)
    parser.add_argument("--checkpoint", type=str, default='./weights/droid.pth', help="checkpoint file")
    parser.add_argument("--global-ba-frontend", type=int, default=0, help="frequency to run global ba on frontend")
    args = parser.parse_args()

    main(
        input_images=args.images,
        input_depth=args.depth,
        intrinsic=args.intr,
        viz=args.viz,
        output_traj=args.out_traj,
        output_poses=args.out_poses,
        output_pcd=args.out_pcd,
        checkpoint=args.checkpoint,
        global_ba_frontend=args.global_ba_frontend
    )
