# standard library
from pathlib import Path
from typing import *
import os, shutil
# thrid party
import argparse
from tqdm import tqdm
import cv2
import numpy as np
# modules
import depth, slam, mesh
from modules.utils import sample_from_video


def data_preprocess(args):
    os.makedirs(args.output, exist_ok=True)

    # video sample
    if not args.input.is_dir():
        # out dir for images
        output_rgb = args.output / 'rgb'
        os.makedirs(output_rgb, exist_ok=True)

        sample_from_video(
            video_path=args.input,
            output_dir=output_rgb,
            sample_fps=args.sample_fps,
        )

        args.input = output_rgb
    
    if args.intr is None:
        # load one image
        img = cv2.imread(str(args.input / os.listdir(args.input)[0]))
        
        # estimate intrinsic
        h, w = img.shape[:2]
        f = np.max([w, h]) * 1.2
        intrinsic = np.array([f, f, w / 2, h / 2])
        
        # save
        np.savetxt(args.output / 'intrinsic.txt', intrinsic)
        args.intr = args.output / 'intrinsic.txt'


def reconstruct_from_images(args):
    depth.main(
        input_images=args.input,
        output_dir=args.output / 'depth',
        intrinsic=args.intr,
        overwrite=not args.skip_existed,
    )

    slam.main(
        input_images=args.input,
        input_depth=args.output / 'depth',
        intrinsic=args.intr,
        viz=args.viz,
        output_poses=args.output / 'poses',
        output_pcd=args.output / 'pcd'
    )

    mesh.main(
        input_images=args.input,
        input_depth=args.output / 'depth',
        input_poses=args.output / 'poses',
        intrinsic=args.intr,
        output_mesh=args.output / 'mesh_simple.ply',
        voxel_length=args.voxel_length,
        smp_decimation=args.smp_decimation,
        smp_voxel_length=args.smp_voxel_length,
        smp_smooth_iter=args.smp_smooth_iter
    )
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running Reconstruction')

    # data
    parser.add_argument('--input', type=str, required=True, help='path for test video or image sequence')
    parser.add_argument('--output', type=str, required=True, help='output directory')
    parser.add_argument('--intr', type=str, help='camera intrinsic file (4 elements)', default=None)
    parser.add_argument('--viz', action='store_true', help='visualize', default=False)
    parser.add_argument('--skip-existed', action='store_true', help='skip existing depth file', default=False)
    parser.add_argument('--sample-fps', type=int, default=60, help='sample FPS for video')

    # checkpoints
    parser.add_argument("--metric-ckpt", type=str, default='./weights/metric_depth_vit_giant2_800k.pth', help='metric3d checkpoint file')
    parser.add_argument("--model-name", type=str, default='v2-g', choices=['v2-L', 'v2-S', 'v2-g'], help='metric 3d model type')
    parser.add_argument("--droid-ckpt", type=str, default='./weights/droid.pth', help="droid-slam checkpoint file")

    # mesh reconstruction
    parser.add_argument("--voxel-length", type=float, default=0.02, help="voxel length")
    parser.add_argument("--smp-decimation", type=int, default=0, help="target_number_of_triangles")
    parser.add_argument("--smp-voxel-length", type=float, default=None, help="voxel length for simplification")
    parser.add_argument("--smp-smooth-iter", type=int, default=40, help="number of smoothing iterations")

    args = parser.parse_args()

    args.input = Path(args.input).resolve()
    args.output = Path(args.output).resolve()

    data_preprocess(args)
    reconstruct_from_images(args)


    # # depth estimation
    # intrinsic = np.loadtxt(args.intr)[:4]
