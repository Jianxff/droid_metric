# standard library
from typing import *
from pathlib import Path
# third party
import argparse
import numpy as np
# RGBD Fusion
from modules import RGBDFusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True, help="directory to rgb images")
    parser.add_argument("--depth", type=str, required=True, help="depth directory")
    parser.add_argument("--traj", type=str, required=True, help="trajectory file")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="depth scale factor")
    parser.add_argument("--viz", type=bool, default=False, help="visualize")
    parser.add_argument("--focal", type=float, default=None, help="focal length")
    parser.add_argument("--calib", type=str, default=None, help="calib file, overwrite focal")
    parser.add_argument("--mesh", type=str, default=None, help="save mesh", required=True)
    parser.add_argument("--voxel_length", type=float, default=0.05, help="voxel length")
    # simply
    parser.add_argument("--smp_decimation", type=int, default=6500, help="target_number_of_triangles")
    parser.add_argument("--smp_voxel_length", type=float, default=None, help="voxel length for simplification")
    parser.add_argument("--smooth_iter", type=int, default=100, help="number of smoothing iterations")

    args = parser.parse_args()

    intr, distort = None, None
    if args.calib:
        calib = np.loadtxt(args.calib)
        intr = calib[:4]
        if len(calib) > 4:
            distort = calib[4:]
    elif args.focal:
        intr = args.focal
    
    mesh = RGBDFusion.pipeline(
        image_dir=args.rgb,
        depth_dir=args.depth,
        traj_dir=args.traj,
        intrinsic=intr,
        distort=distort,
        viz=args.viz,
        voxel_length=args.voxel_length,
        mesh_save=Path(args.mesh).parent / 'mesh_raw.ply'
    )
    
    RGBDFusion.simplify_mesh(
        mesh=mesh,
        save=args.mesh,
        decimation=args.smp_decimation,
        voxel_size=args.smp_voxel_length,
        smooth_iter=args.smooth_iter
    )