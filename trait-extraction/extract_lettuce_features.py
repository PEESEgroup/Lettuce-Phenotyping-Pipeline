import numpy as np
import cv2

def extract_lettuce_features(masked_clouds, labeled_object):
    W1 = []
    W2 = []
    heights = []

    for i in range(len(masked_clouds)):
        try:
            masked_cloud_i = masked_clouds[i]
            curr_lettuce = masked_cloud_i[labeled_object[i] == 1]
            depth_values = curr_lettuce[:, 2]
            height = (np.max(depth_values) - np.min(depth_values)) * 100  # in cm

            W1.append(width)
            W2.append(height_bbox)
            heights.append(height)

        except RuntimeError as e:
            print(f"Iteration {i}: Error - {e}")
            continue

    return W1, W2, heights

def main(masked_clouds, labeled_object):
    W1, W2, heights = extract_lettuce_features(masked_clouds, labeled_object)
    return W1, W2, heights

if __name__ == "__main__":
    masked_clouds = np.load('path_to_masked_clouds.npy')  # Load your data
    labeled_object = np.load('path_to_labeled_object.npy')  # Load labeled object
    W1, W2, heights = main(masked_clouds, labeled_object)
