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
# droid_slam_path = Path(__file__).resolve().parent / 'droid_slam/droid_slam'
from .droid_core.droid import Droid
from .data import PosedImageStream

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
    convert_pose_gl: bool = False
    poses_dir: Path = None
    distort: np.ndarray = None
    global_ba_frontend: int = 0

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


class RGBDStream(PosedImageStream):
    def __init__(
        self,
        image_dir: Path,
        depth_dir: Optional[Path],
        stride: Optional[int] = 1,
        intrinsic: Optional[Union[float, np.ndarray]] = None,
        distort: Optional[np.ndarray] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            image_dir=image_dir,
            depth_dir=depth_dir,
            stride=stride,
            intrinsic=intrinsic,
            distort=distort,
            resize=resize
        )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb, depth, _, intr = super().__getitem__(idx)
        h1, w1 = rgb.shape[:2]

        rgb = rgb[:h1-h1%8, :w1-w1%8]
        if depth is not None: depth = depth[:h1-h1%8, :w1-w1%8]

        rgb = torch.as_tensor(rgb).permute(2, 0, 1)
        if depth is not None: depth = torch.as_tensor(depth)
        intr = torch.as_tensor(intr)

        return rgb[None], depth, intr

        
def run(
    image_dir: Path,
    setting: Optional[Options] = Options(),
    depth_dir: Optional[Path] = None
) -> np.ndarray:
    """ main function """

    droid: Droid = None

    torch.multiprocessing.set_start_method('spawn', force=True)

    keyframe_watcher = 0

    dataset = RGBDStream(
        image_dir=image_dir,
        depth_dir=depth_dir,
        stride=setting.stride,
        intrinsic=setting.intrinsic if setting.intrinsic is not None else setting.focal,
        distort=setting.distort,
        resize=(512, 384)
    )

    for t, (image, depth, intr) in tqdm(enumerate(dataset)):
        if t < setting.t0:
            continue
        # show image if visualize
        if not setting.disable_vis:
            show_image(image[0])
        # create droid instance if None
        if droid is None:
            setting.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(setting)
        
        # front end
        droid.track(tstamp=t, image=image, depth=depth, intrinsics=intr)
        
        # check keyframe and run global-ba
        keyframes = droid.video.counter.value
        if keyframes != keyframe_watcher:
            keyframe_watcher = keyframes
            if setting.global_ba_frontend > 0 and keyframes >= np.min([3, setting.global_ba_frontend]):
                if keyframes % setting.global_ba_frontend == 0:    
                    droid.backend(7)

    # save reconstruction
    # if setting.reconstruction_path is not None:
    #     droid.save(setting.reconstruction_path)
    
    # fill non-keyframe pose
    def extract_rgb_stream(stream: RGBDStream):
        for t, (im, _, intr) in enumerate(stream):
            yield t, im, intr
    traj_est = droid.terminate(extract_rgb_stream(dataset))

    # save raw trajectory under opencv coordinate
    if setting.trajectory_path is not None:
        np.savetxt(str(setting.trajectory_path), traj_est)

    # save pose44 matrix under opencv/opengl coordinate, ordered by frame
    if setting.poses_dir is not None:
        from .utils import trajectory_to_poses
        trajectory_to_poses(traj_est, setting.poses_dir, convert_opengl=setting.convert_pose_gl)
    
    print('finished')