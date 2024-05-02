# standard library
from pathlib import Path
from typing import *
# thirdparty
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# utils
from .utils import K_from_intr

class PosedImageStream(Dataset):
    def __init__(
        self,
        image_dir: Path,
        depth_dir: Optional[Path] = None,
        traje_dir: Optional[Path] = None,
        stride: Optional[int] = 1,
        intrinsic: Optional[Union[float, np.ndarray]] = None,
        distort: Optional[np.ndarray] = None,
        resize: Optional[Tuple[int, int]] = None,
    ):
        self.distort = distort
        self.stride = stride

        self.rgb_list = \
            sorted(list(Path(image_dir).glob('*.[p|j][n|p]g')))[::stride]
        self.depth_list = None if not depth_dir \
            else sorted(list(Path(depth_dir).glob('*.npy')))[::stride]
        self.pose_list = None if not traje_dir \
            else sorted(list(Path(traje_dir).glob('*.txt')))[::stride]

        w0, h0 = Image.open(self.rgb_list[0]).size
        self.image_size = (w0, h0)
        
        if intrinsic is None or isinstance(intrinsic, float):
            focal = intrinsic if intrinsic is not None else\
                np.max([h0, w0])
            self.intrinsic = np.array([focal, focal, w0 / 2, h0 / 2])
        elif isinstance(intrinsic, np.ndarray):
            self.intrinsic = np.array(intrinsic)
        else:
            raise ValueError("intrinsic must be either None, float or np.ndarray")
        
        self.K_origin = K_from_intr(intr=self.intrinsic)

        # resize
        if resize:
            wr, hr = resize
            h1 = int(h0 * np.sqrt((wr * hr) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((wr * hr) / (h0 * w0)))
            self.intrinsic[0::2] *= (w1 / w0)
            self.intrinsic[1::2] *= (h1 / h0)
            self.resize = (w1, h1)
            self.image_size = (w1, h1)
        else:
            self.resize = None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.distort is not None:
            frame = cv2.undistort(frame, self.K_origin, self.distort)
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        return frame

    def __len__(self) -> int:
        return len(self.rgb_list)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_rgb = cv2.imread(str(self.rgb_list[idx]), cv2.IMREAD_COLOR).astype(np.uint8)
        raw_depth = None if not self.depth_list else np.load(self.depth_list[idx])
        # pack data
        rgb = self.preprocess(raw_rgb)
        depth = None if not self.depth_list else self.preprocess(raw_depth)
        pose = None if not self.pose_list else np.loadtxt(self.pose_list[idx])

        return rgb, depth, pose, self.intrinsic
    

