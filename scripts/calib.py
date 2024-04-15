import argparse
from module import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="images dir or video file for calibration")
    parser.add_argument("pattern", type=str, required=True, choices=['circle', 'chessboard'], help="pattern type: circle, chessboard")
    parser.add_argument("--pattern-size", type=int, nargs=2, required=True, help="pattern size")
    parser.add_argument("--square-size", type=float, default=15, help="square size in millimeter")
    parser.add_argument("--write", type=str, default='calib.txt', help="output file")
    args = parser.parse_args()

    _, intr, dist = utils.calibrate_camera(args.input, args.pattern_size, args.pattern, args.square_size)

    with open(args.write, 'w') as f:
        f.write(f"camera_matrix:\n{intr}\n")
        f.write(f"distortion_coefficients:\n{dist}\n")
        
