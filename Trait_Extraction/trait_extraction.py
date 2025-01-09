#!/usr/bin/env python3

import os
import numpy as np
import imageio.v2 as imageio
import cv2
import pandas as pd
import torch
from plantcv.plantcv._helpers import _object_composition
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

rgb_folder = r"/SAMPLE_RGB" # Replace with your file path
depth_folder = r"/SAMPLE_DEPTH" # Replace with your file path
out_dir = r"/output" # Replace with your file path

sam_checkpoint = r"/sam_vit_h_4b8939.pth" # Replace with your file path
model_type = "vit_h"

fx_d, fy_d, cx_d, cy_d = 388.022, 388.022, 321.261, 229.877

coords = [
    (32, 134), (32, 220), (33, 306), (32, 393), (77, 91),
    (121, 133), (79, 179), (121, 219), (78, 265), (121, 303),
    (77, 352), (123, 395), (184, 111), (185, 199), (187, 286),
    (185, 374), (269, 113), (275, 202), (276, 286), (277, 372),
    (363, 111), (363, 199), (361, 283), (363, 376), (452, 111),
    (453, 199), (453, 283), (454, 374), (514, 87),  (559, 135),
    (519, 178), (565, 220), (520, 263), (572, 311), (523, 356),
    (568, 403), (613, 87),  (612, 176), (609, 263), (614, 355)
]

def create_aligned_array(depth_data, rgb_data, fx, fy, cx, cy):
    aligned = np.zeros((depth_data.shape[0], depth_data.shape[1], 6), dtype=np.float32)
    for v in range(depth_data.shape[0]):
        for u in range(depth_data.shape[1]):
            z = float(depth_data[v, u]) * 0.001
            x = ((u - cx) * z) / (2 * fx)
            y = ((v - cy) * z) / (2 * fy)
            aligned[v, u, :3] = [x, y, z]
            aligned[v, u, 3:] = rgb_data[v, u, :]
    return aligned

def parse_timestamp_from_filename(filename):
    base = os.path.splitext(filename)[0]
    if base.startswith("image_"):
        base = base.replace("image_", "", 1)
    return base

def main():
    os.makedirs(out_dir, exist_ok=True)

    w1_cols = ["timestamp"] + [f"W1_{i}" for i in range(40)]
    w2_cols = ["timestamp"] + [f"W2_{i}" for i in range(40)]
    maxz_cols = ["timestamp"] + [f"MaxZ_{i}" for i in range(40)]
    minz_cols = ["timestamp"] + [f"MinZ_{i}" for i in range(40)]

    df_w1 = pd.DataFrame(columns=w1_cols)
    df_w2 = pd.DataFrame(columns=w2_cols)
    df_maxz = pd.DataFrame(columns=maxz_cols)
    df_minz = pd.DataFrame(columns=minz_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    rgb_files = sorted([
        f for f in os.listdir(rgb_folder)
        if f.startswith("image_") and f.endswith(".png")
    ])

    for idx, rgb_file in enumerate(tqdm(rgb_files, desc="Processing images", unit="pair"), start=1):
        timestamp_str = parse_timestamp_from_filename(rgb_file)
        depth_file = f"depthimg_{timestamp_str}.png"

        rgb_path = os.path.join(rgb_folder, rgb_file)
        depth_path = os.path.join(depth_folder, depth_file)

        if not os.path.exists(depth_path):
            print(f"[WARNING] Depth file not found for {rgb_file}. Skipping...")
            continue

        rgb_data = imageio.imread(rgb_path)
        depth_data = imageio.imread(depth_path)

        if rgb_data.shape[:2] != depth_data.shape[:2]:
            print(f"[WARNING] Mismatched dimensions for {rgb_file} & {depth_file}. Skipping...")
            continue

        aligned = create_aligned_array(depth_data, rgb_data, fx_d, fy_d, cx_d, cy_d)
        img_for_sam = cv2.cvtColor(aligned[:, :, 3:6].astype(np.uint8), cv2.COLOR_BGR2RGB)
        predictor.set_image(img_for_sam)

        labeled_masks = []
        for (ptx, pty) in coords:
            input_point = np.array([[ptx, pty]])
            input_label = np.array([1])
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            labeled_masks.append(masks[best_idx])

        W1_cm = []
        W2_cm = []
        MaxZ_cm = []
        MinZ_cm = []

        for mask_i in labeled_masks:
            masked_cloud = aligned.copy()
            masked_cloud[mask_i == 0] = 0

            bgr_img = cv2.cvtColor(masked_cloud[:, :, 3:6].astype(np.uint8), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            _, bin_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                W1_cm.append(0.0)
                W2_cm.append(0.0)
                MaxZ_cm.append(0.0)
                MinZ_cm.append(0.0)
                continue

            obj = _object_composition(contours=contours, hierarchy=hierarchy)
            x, y, w_bbox, h_bbox = cv2.boundingRect(obj)
            region_3D = aligned[y : y + h_bbox, x : x + w_bbox, :3]
            valid_pts = region_3D[region_3D[..., 2] > 0]

            if valid_pts.size == 0:
                W1_cm.append(0.0)
                W2_cm.append(0.0)
                MaxZ_cm.append(0.0)
                MinZ_cm.append(0.0)
                continue

            min_x, max_x = np.min(valid_pts[:, 0]), np.max(valid_pts[:, 0])
            min_y, max_y = np.min(valid_pts[:, 1]), np.max(valid_pts[:, 1])
            min_z, max_z = np.min(valid_pts[:, 2]), np.max(valid_pts[:, 2])

            W1_cm.append((max_x - min_x) * 100.0)
            W2_cm.append((max_y - min_y) * 100.0)
            MaxZ_cm.append(max_z * 100.0)
            MinZ_cm.append(min_z * 100.0)

        row_w1 = {"timestamp": timestamp_str}
        row_w2 = {"timestamp": timestamp_str}
        row_maxz = {"timestamp": timestamp_str}
        row_minz = {"timestamp": timestamp_str}

        for i_mask in range(40):
            row_w1[f"W1_{i_mask}"] = W1_cm[i_mask] if i_mask < len(W1_cm) else 0.0
            row_w2[f"W2_{i_mask}"] = W2_cm[i_mask] if i_mask < len(W2_cm) else 0.0
            row_maxz[f"MaxZ_{i_mask}"] = MaxZ_cm[i_mask] if i_mask < len(MaxZ_cm) else 0.0
            row_minz[f"MinZ_{i_mask}"] = MinZ_cm[i_mask] if i_mask < len(MinZ_cm) else 0.0

        df_w1.loc[len(df_w1)] = row_w1
        df_w2.loc[len(df_w2)] = row_w2
        df_maxz.loc[len(df_maxz)] = row_maxz
        df_minz.loc[len(df_minz)] = row_minz

        print(f"[INFO] Processed {idx}/{len(rgb_files)} -> {rgb_file}")

    df_w1.to_csv(os.path.join(out_dir, "W1.csv"), index=False)
    df_w2.to_csv(os.path.join(out_dir, "W2.csv"), index=False)
    df_maxz.to_csv(os.path.join(out_dir, "max_z.csv"), index=False)
    df_minz.to_csv(os.path.join(out_dir, "min_z.csv"), index=False)
    print("[INFO] Done! Files saved in:", out_dir)

if __name__ == "__main__":
    main()

