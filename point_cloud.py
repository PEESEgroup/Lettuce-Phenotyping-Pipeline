import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def depth_rgb_registration(depth_data, rgb_data, fx_d, fy_d, cx_d, cy_d, fx_rgb, fy_rgb, cx_rgb, cy_rgb, extrinsics):
    '''
    Perform depth and RGB data registration using provided intrinsics and extrinsics.

    depth_data: the depth data
    rgb_data:  the RGB data
    fx_d: focal length in x-direction for depth camera
    fy_d: focal length in y-direction for depth camera
    cx_d: principal point in x-direction for depth camera
    cy_d: principal point in y-direction for depth camera
    fx_rgb: focal length in x-direction for RGB camera
    fy_rgb: focal length in y-direction for RGB camera
    cx_rgb: principal point in x-direction for RGB camera
    cy_rgb: principal point in y-direction for RGB camera
    extrinsics: the extrinsics matrix

    aligned: the aligned data containing X, Y, Z, R, G, B values
    '''
    depth_height, depth_width = depth_data.shape[:2]
    rgb_height, rgb_width = rgb_data.shape[:2]

    # Aligned contains X, Y, Z, R, G, B values
    aligned = np.zeros((depth_height, depth_width, 6), dtype=np.float32)

    for v in range(depth_height):
        for u in range(depth_width):
            # Depth intrinsics
            z = float(depth_data[v, u]) * 0.001
            x = float((u - cx_d) * z) / fx_d
            y = float((v - cy_d) * z) / fy_d

            # Extrinsics
            transformed = np.dot(extrinsics, np.array([x, y, z, 1]))
            aligned[v, u, :3] = transformed[:3]

    for v in range(depth_height):
        for u in range(depth_width):
            # RGB intrinsics
            x = (aligned[v,u,0] * fx_rgb / aligned[v,u,2]) + cx_rgb
            y = (aligned[v,u,1] * fy_rgb / aligned[v,u,2]) + cy_rgb

            # If x,y are valid indices
            if (x >= rgb_width or y >= rgb_height or
                x < 0 or y < 0 or
                np.isnan(x) or np.isnan(y)):
                continue

            # Round indices
            x, y = int(round(x)), int(round(y))
            if (x >= rgb_width or y >= rgb_height or
                x < 0 or y < 0):
                continue

            aligned[v, u, 3:] = rgb_data[y, x, :].astype(np.float32)
            
    return aligned

def plot_crop_geometry(aligned):
    '''
    Generate a 3D scatter plot of the input aligned point cloud data.

    aligned: The aligned point cloud data containing 3D coordinates and color values.
    '''
    aligned_point_cloud = aligned
    
    # Extract 3D coordinates and color values
    xyz = aligned_point_cloud[:, :, :3]
    rgb = aligned_point_cloud[:, :, 3:]

    points = xyz.reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0 

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=-145, azim=-90)
    ax.grid(False)    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    zoom_factor = 0.5 
    x_center, y_center, z_center = np.mean(points, axis=0)
    x_range, y_range, z_range = np.ptp(points, axis=0) * zoom_factor

    ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
    ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
    ax.set_zlim(z_center - z_range/2, z_center + z_range/2)
    plt.show()