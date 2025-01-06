import numpy as np
import imageio.v2 as imageio

def create_aligned_array(depth_data, rgb_data, fx_d, fy_d, cx_d, cy_d):
    depth_height, depth_width = depth_data.shape[:2]
    aligned = np.zeros((depth_height, depth_width, 6), dtype=np.float32)

    for v in range(depth_height):
        for u in range(depth_width):
            z = float(depth_data[v, u]) * 0.001  # Assuming depth is in millimeters
            x = float((u - cx_d) * z) / (2* fx_d)
            y = float((v - cy_d) * z) / (2* fy_d)
            aligned[v, u, :3] = [x, y, z]
            aligned[v, u, 3:] = rgb_data[v, u, :]

    return aligned

def load_images(depth_path, rgb_path):
    depth_data = imageio.imread(depth_path)
    rgb_data = imageio.imread(rgb_path)
    return depth_data, rgb_data

def main(depth_path, rgb_path, fx_d, fy_d, cx_d, cy_d):
    depth_data, rgb_data = load_images(depth_path, rgb_path)
    aligned = create_aligned_array(depth_data, rgb_data, fx_d, fy_d, cx_d, cy_d)
    return aligned

if __name__ == "__main__":
    depth_path = "path_to_depth_image"
    rgb_path = "path_to_rgb_image"
    fx_d, fy_d, cx_d, cy_d = 388.022, 388.022, 321.261, 229.877  # Example values
    aligned = main(depth_path, rgb_path, fx_d, fy_d, cx_d, cy_d)
