import numpy as np
import cv2
from segment_anything import SamPredictor

def process_masks(coords, predictor, img):
    labeled_object = []
    for x, y in coords:
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
        highest_score_index = np.argmax(scores)
        labeled_object.append(masks[highest_score_index])

    return labeled_object

def combine_masks(labeled_object, img_shape):
    combined_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    for mask in labeled_object:
        mask_resized = cv2.resize(mask.astype(np.uint8), (img_shape[1], img_shape[0]))
        combined_mask = np.maximum(combined_mask, mask_resized)

    return combined_mask

def main(coords, predictor, img):
    labeled_object = process_masks(coords, predictor, img)
    combined_mask = combine_masks(labeled_object, img.shape)
    return labeled_object, combined_mask

if __name__ == "__main__":
    # Example usage with loaded images and predictor
    predictor = SamPredictor(model="path_to_model_checkpoint")
    coords = [(32, 134), (32, 220), (33, 306)]  # Example coordinates
    img = np.load('path_to_image.npy')  # Load image data
    labeled_object, combined_mask = main(coords, predictor, img)
