import open3d as o3d
import numpy as np

def create_point_cloud_from_aligned(aligned):
    xyz = aligned[:, :, :3]
    rgb = aligned[:, :, 3:]
    points = xyz.reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0  # Assuming RGB values are in the range [0, 255]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])

def main(aligned):
    point_cloud = create_point_cloud_from_aligned(aligned)
    visualize_point_cloud(point_cloud)

if __name__ == "__main__":
    aligned = np.load('path_to_aligned_array.npy')  # Or pass the aligned data directly
    main(aligned)
