import open3d as o3d

def demo_crop_geometry(point_cloud):
    o3d.visualization.draw_geometries_with_editing([point_cloud])

if __name__ == "__main__":
    point_cloud = o3d.geometry.PointCloud()  # Load or create a point cloud
    demo_crop_geometry(point_cloud)
