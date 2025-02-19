# -*- coding: utf-8 -*-
"""data_capture.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_h40D2iU6QNQOczaODKwFLaTm20-vNqy
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # Configuration
    output_folder = "captured_images"  # Main output directory
    rgb_folder = os.path.join(output_folder, "rgb_images")       # RGB images folder
    depth_folder = os.path.join(output_folder, "depth_images")   # Depth images folder
    create_folder(rgb_folder)
    create_folder(depth_folder)

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure the pipeline to stream depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Create an align object to align depth frames to color frames
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        print("Streaming started. Press Ctrl+C to stop.")
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth or not color_frame:
                print("Could not acquire both depth and color frames.")
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Optional: Apply colormap on depth image for visualization
            # Uncomment the following lines if you wish to save color-mapped depth images
            # depth_colormap = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_image, alpha=0.03),
            #     cv2.COLORMAP_JET
            # )

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y-%m-%d %H_%M_%S")

            # Define file paths with desired formats
            rgb_filename = os.path.join(rgb_folder, f"image_{timestamp}.png")
            depth_filename = os.path.join(depth_folder, f"depthimg_{timestamp}.png")

            # Save the images
            cv2.imwrite(rgb_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)  # Save raw depth image
            # To save the color-mapped depth image, use the following line instead:
            # cv2.imwrite(depth_filename, depth_colormap)

            print(f"Saved RGB image as {rgb_filename}")
            print(f"Saved Depth image as {depth_filename}")

            # Wait for 10 minutes (600 seconds)
            time.sleep(600)

    except KeyboardInterrupt:
        print("\nStopping the script...")

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    main()