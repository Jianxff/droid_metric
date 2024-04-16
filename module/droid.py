# standard library
from pathlib import Path
from typing import *
import sys
# third party
import cv2
import numpy as np
import torch
from tqdm import tqdm
# droid slam
droid_slam_path = Path(__file__).resolve().parent / 'droid_slam/droid_slam'
sys.path.append(str(droid_slam_path))
from .droid_slam.droid_slam.droid import Droid

__ALL__ = ['run', 'Options']

class Options:
    image_size: np.ndarray = None
    weights: Path = Path('weights/droid.pth')
    stereo: bool = False
    t0: int = 0
    stride: int = 1
    buffer: int = 512
    disable_vis: bool = True
    beta: float = 0.3
    warmup: int = 8
    filter_thresh: float = 2.4
    keyframe_thresh: float = 4.0
    frontend_thresh: float = 16.0
    frontend_window: int = 25
    frontend_radius: int = 2
    frontend_nms: int = 1
    backend_thresh: float = 22.0
    backend_radius: int = 2
    backend_nms: int = 3
    upsample: bool = False
    reconstruction_path: Path = None
    # new options
    intrinsic: np.ndarray = None
    focal: float = None
    trajectory_path: Path = None
    poses_dir: Path = None
    depth_scale: float = 1.0
    distort: np.ndarray = None
    global_ba_frontend: int = 0

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(image_dir: Path, setting: Options, depth_dir: Path = None):
    """ image generator """
    stride = setting.stride
    focal = setting.focal
    depth_scale = setting.depth_scale
    distort = setting.distort

    image_list = sorted(Path(image_dir).glob('*.[p|j][n|p]g'))[::stride]
    first_image = cv2.imread(str(image_list[0]))

    # depth
    use_depth = depth_dir is not None
    depth_list = [] if not use_depth else sorted(Path(depth_dir).glob('*.png'))[::stride]
    if use_depth:
        first_depth = cv2.imread(str(depth_list[0]), cv2.IMREAD_UNCHANGED)
        assert first_image.shape[:2] == first_depth.shape[:2], \
            "depth and image size mismatch"

    # calculate intrinsic
    K = np.eye(3)
    if setting.intrinsic is None:
        h, w = first_image.shape[:2]
        if focal is None: focal = np.max([h, w]) # predict focal length
        cx, cy = w / 2, h / 2    
        K[0, 0] = K[1, 1] = focal
        K[0, 2], K[1, 2] = cx, cy
        intrinsic = torch.as_tensor([focal, focal, cx, cy])
    else:
        intrinsic = setting.intrinsic
        K[0, 0], K[1, 1] = intrinsic[0], intrinsic[1] # fx, fy
        K[0, 2], K[1, 2] = intrinsic[2], intrinsic[3] # cx, cy
        intrinsic = torch.as_tensor(setting.intrinsic)

    # resize intrinsic
    h0, w0, _ = first_image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    intrinsic[0::2] *= (w1 / w0)
    intrinsic[1::2] *= (h1 / h0)

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        # distortion
        if distort is not None:
            image = cv2.undistort(image, K, distort)      
        # resize
        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        if use_depth:
            depth = cv2.imread(str(depth_list[t]), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / depth_scale
            if distort is not None:
                depth = cv2.undistort(depth, K, distort)
            depth = cv2.resize(depth, (w1, h1))
            depth = depth[:h1-h1%8, :w1-w1%8]
            depth = torch.as_tensor(depth)

            yield t, (image[None], depth), intrinsic
        
        else:
            yield t, image[None], intrinsic
        
def run(
    image_dir: Path,
    setting: Optional[Options] = Options(),
    depth_dir: Optional[Path] = None
) -> np.ndarray:
    """ main function """

    droid: Droid = None

    torch.multiprocessing.set_start_method('spawn')

    keyframe_watcher = 0

    for (t, data, intrinsic) in tqdm(image_stream(image_dir, setting, depth_dir)):
        if t < setting.t0:
            continue
        # check depth data
        if depth_dir is not None:
            image, depth = data
        else:
            image, depth = data, None
        # show image if visualize
        if not setting.disable_vis:
            show_image(image[0])
        # create droid instance if None
        if droid is None:
            setting.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(setting)
        
        # front end
        droid.track(tstamp=t, image=image, depth=depth, intrinsics=intrinsic)
        
        # check keyframe and run global-ba
        keyframes = droid.video.counter.value
        if keyframes != keyframe_watcher:
            keyframe_watcher = keyframes
            if setting.global_ba_frontend > 0 and keyframes >= np.min([3, setting.global_ba_frontend]):
                if keyframes % setting.global_ba_frontend == 0:    
                    droid.backend()

    
    if setting.reconstruction_path is not None:
        droid.save(setting.reconstruction_path)
    
    traj_est = droid.terminate(image_stream(image_dir, setting))

    if setting.trajectory_path is not None:
        np.savetxt(setting.trajectory_path, traj_est)

    if setting.poses_dir is not None:
        from .utils import trajecitry_to_poses
        trajecitry_to_poses(traj_est, setting.poses_dir)
    
    print('finished')