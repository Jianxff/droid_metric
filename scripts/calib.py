import argparse
import numpy as np
from modules import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="images dir or video file for calibration")
    parser.add_argument("--pattern", type=str, required=True, choices=['circle', 'chessboard'], help="pattern type: circle, chessboard")
    parser.add_argument("--pattern-size", type=int, nargs=2, required=True, help="pattern size")
    parser.add_argument("--square-size", type=float, default=15, help="square size in millimeter")
    parser.add_argument("--write", type=str, default='calib.txt', help="output file")
    args = parser.parse_args()

    print(f'pattern size: {args.pattern_size}')

    if args.pattern == 'circle':
        args.pattern_size = (args.pattern_size[1], args.pattern_size[0])

    _, intr, dist = utils.calibrate_camera(
        images=args.input, 
        pattern_type=args.pattern, 
        pattern_size=args.pattern_size,
        square_size=args.square_size)

    if args.write:
        calib = np.hstack([intr, dist])
        np.savetxt(args.write, calib)

