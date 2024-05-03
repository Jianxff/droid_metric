import argparse
import open3d as o3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='scene file')
    parser.add_argument('--traj', type=str, help='trajectory file')
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.scene)
    lineset = o3d.io.read_line_set(args.traj)
    # paint lineset
    lineset.paint_uniform_color([1, 0, 0])

    # visualize
    o3d.visualization.draw_geometries([pcd, lineset])