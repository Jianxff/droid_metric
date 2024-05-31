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
    parser.add_argument("--poses", type=str, required=True, help="trajectory file")
    parser.add_argument("--viz", type=bool, default=False, help="visualize")
    parser.add_argument("--intr", type=str, default=None, help="intrinsic file")
    parser.add_argument("--save", type=str, default=None, help="save mesh", required=True)
    parser.add_argument("--voxel_length", type=float, default=0.02, help="voxel length")
    # simplify
    parser.add_argument("--smp_decimation", type=int, default=0, help="target_number_of_triangles")
    parser.add_argument("--smp_voxel_length", type=float, default=None, help="voxel length for simplification")
    parser.add_argument("--smooth_iter", type=int, default=40, help="number of smoothing iterations")

    args = parser.parse_args()

    intr = np.loadtxt(args.intr)[:4]
    
    mesh = RGBDFusion.pipeline(
        image_dir=args.rgb,
        depth_dir=args.depth,
        traj_dir=args.poses,
        intrinsic=intr,
        viz=args.viz,
        voxel_length=args.voxel_length,
        mesh_save=Path(args.save).parent / 'mesh_raw.ply'
    )
    
    RGBDFusion.simplify_mesh(
        mesh=mesh,
        save=args.save,
        decimation=args.smp_decimation,
        voxel_size=args.smp_voxel_length,
        smooth_iter=args.smooth_iter
    )