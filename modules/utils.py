# standard library
from pathlib import Path
from typing import *
import time
import os, shutil
# thirdparty
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from lietorch import SO3
import torch


def sample_from_video(
    video_path: Union[Path, str],
    output_dir: Union[Path, str],
    sample_fps: float = 30,
    limit: Optional[int] = None
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video file: {video_path}')
    
    os.makedirs(str(output_dir), exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = np.max([fps // sample_fps, 1])

    if limit is not None:
        stride = 1 + cnt // limit
        sample_rate = np.max([sample_rate, stride])
    
    print(f'sample from video {video_path} with stride {sample_rate}')
    for i in tqdm(range(cnt)):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % sample_rate == 0:
            cv2.imwrite(str(output_dir / f'{i:06d}.png'), frame)
    cap.release()


def calibrate_camera(
    images: Union[str, Path, List[Union[str, Path, np.ndarray, Image.Image]]],
    pattern_type: str, # 'chessboard' or 'circle'
    pattern_size: Tuple[int, int], # (rows, cols)
    square_size: float = 15.0,
    image_limit: int = 150
) -> Tuple[float, np.ndarray, np.ndarray]: # Intrinsics, Distortion
    assert pattern_type in ['chessboard', 'circle'], \
        f'pattern_type should be either chessboard or circle, got {pattern_type}'

    del_cache = None
    if isinstance(images, (str, Path)):
        if Path(images).suffix in ['.mp4', '.avi']:
            cache_dir = Path(images).parent / f'.calib.{int(time.time() * 1e9)}'
            del_cache = cache_dir
            sample_from_video(video_path=images, output_dir=cache_dir, limit=image_limit)
        elif Path(images).is_dir():
            cache_dir = Path(images)
        else:
            raise ValueError(f'images should be either video file or directory, got {images}')
        images = sorted(list(cache_dir.glob('*.[p|j][n|p]g')))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows, cols = pattern_size

    objp = np.zeros((rows * cols ,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
    objp *= square_size

    obj_points = []
    img_points = []

    print('detecting chessboard' if pattern_type == 'chessboard' else 'detecting circle')
    for image in tqdm(images):
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        ret, corners = \
            cv2.findChessboardCorners(gray, pattern_size, None) if pattern_type == 'chessboard' else \
            cv2.findCirclesGrid(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)
            if pattern_type == 'chessboard':
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                img_points.append(corners2 if [corners2] else corners)
            else:
                img_points.append(corners)

    print(f'grab {len(img_points)} images for calibration')

    # calibrate
    print('run calibration')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    print('reprojection error:', ret)
    print(f'intrinsic: fx={mtx[0][0]}, fy={mtx[1][1]}, cx={mtx[0][2]}, cy={mtx[1][2]}')
    print(f'distortion: k1={dist[0][0]}, k2={dist[0][1]}, p1={dist[0][2]}, p2={dist[0][3]}, k3={dist[0][4]}')
    
    if del_cache is not None:
        shutil.rmtree(str(del_cache), ignore_errors=True)

    intrinsic = np.array([mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]]).flatten()
    distortion = np.array([dist[0][0], dist[0][1], dist[0][2], dist[0][3], dist[0][4]]).flatten()
    

    return (ret, intrinsic, distortion)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    Q = SO3.InitFromVec(torch.Tensor(q))
    R = Q.matrix().detach().cpu().numpy().astype(np.float32)
    return R[:3, :3]


def trajectory_to_poses(
    traj: Union[str, Path, np.ndarray],
    out_dir: Union[str, Path],
) -> None:
    if isinstance(traj, (str, Path)):
        traj = np.loadtxt(traj)
    out_dir = Path(out_dir)

    os.makedirs(str(out_dir), exist_ok=True)
    
    for i in range(len(traj)):
        pose = traj[i]
        t, q = pose[1:4], pose[4:]
        R = quaternion_to_matrix(q)
        T = np.eye(4)
        # Twc = [R | t]
        T[:3, :3] = R
        T[:3, 3] = t
        # write to poses
        np.savetxt(out_dir / f'{i:06d}.txt', T)


def K_from_intr(
    intr: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None
) -> np.ndarray:
    if intr is None:
        intr = np.array([fx, fy, cx, cy])
    K = np.eye(3)
    K[0, 0], K[1, 1] = intr[0], intr[1]
    K[0, 2], K[1, 2] = intr[2], intr[3]
    return K

