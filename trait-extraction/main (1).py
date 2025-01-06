from align_images import main as align_images
from create_point_cloud import main as create_point_cloud
from process_masks import main as process_masks
from extract_lettuce_features import main as extract_features
import numpy as np

def run_pipeline(depth_path, rgb_path, coords, predictor):
    aligned = align_images(depth_path, rgb_path)
    point_cloud = create_point_cloud(aligned)
    labeled_object, combined_mask = process_masks(coords, predictor, point_cloud)
    W1, W2, heights = extract_features(masked_clouds, labeled_object)
    return W1, W2, heights

if __name__ == "__main__":
    depth_path = "path_to_depth_image"
    rgb_path = "path_to_rgb_image"
    coords = [(32, 134), (32, 220), (33, 306), (32, 393), (77, 91),(121, 133), (79, 179), (121, 219), (78, 265), (121, 303), (77, 352), (123, 395), (184, 111), (185, 199), (187, 286), (185, 374), (269, 113), (275, 202), (276, 286), (277, 372), (363, 111),
    (363, 199),
    (361, 283),
    (363, 376),
    (452, 111),
    (453, 199),
    (453, 283),
    (454, 374),
    (514, 87),
    (559, 135),
    (519, 178),
    (565, 220),
    (520, 263),
    (572, 311),
    (523, 356),
    (568, 403),
    (613, 87),
    (612, 176),
    (609, 263),
    (614, 355)]

    predictor = SamPredictor(model="path_to_model_checkpoint")

    W1, W2, heights = run_pipeline(depth_path, rgb_path, coords, predictor)
    print(f"Widths: {W1}")
    print(f"Bounding box widths: {W2}")
    print(f"Heights: {heights}")
