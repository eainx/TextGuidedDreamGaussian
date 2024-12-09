from PIL import Image
import numpy as np
import cv2
import random

def segment_image_old(image, segmentation_mask):
    image_array = np.array(image)
    # Create an RGBA array of zeros (fully transparent) with the same shape as the input image
    transparent_image_array = np.zeros(image_array.shape, dtype=np.uint8)

    # Apply the segmentation mask to copy the relevant parts of the original image
    for channel in range(3):  # Iterate over RGB channels
        transparent_image_array[..., channel][segmentation_mask] = image_array[..., channel][segmentation_mask]

    # Set the alpha channel to 255 (fully opaque) where the segmentation mask is True
    transparent_image_array[..., 3][segmentation_mask] = 255

    # Convert the NumPy array back to an Image
    transparent_image = Image.fromarray(transparent_image_array, mode='RGBA')

    return transparent_image

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    # Create an RGBA array of zeros (fully transparent) with the same shape as the input image
    transparent_image_array = np.zeros(image_array.shape, dtype=np.uint8)

    # Apply the segmentation mask to copy the relevant parts of the original image
    for channel in range(3):  # Iterate over RGB channels
        transparent_image_array[..., channel][segmentation_mask] = image_array[..., channel][segmentation_mask]

    # Set the alpha channel to 255 (fully opaque) where the segmentation mask is True
    transparent_image_array[..., 3][segmentation_mask] = 255

    return transparent_image_array

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

def get_indices_of_max_value(values):
    max_value = max(values)  # Find the maximum value in the list
    return [i for i, v in enumerate(values) if v == max_value]

def image_resize(img, W, H):
    masked_img = np.array(img)

    h, w = masked_img.shape[:2]
    ratio = min(W / w, H / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    # 이미지 리사이즈
    img_resized = cv2.resize(masked_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    new_img = np.zeros((H, W, 4), dtype=np.uint8)
    x_offset = (W - new_w) // 2
    y_offset = (H - new_h) // 2

    new_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    return new_img

def calculate_iou(mask1, mask2):
    # IoU of mask1 (not both)
    intersection = np.logical_and(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(mask1)
    return iou_score

def combine_masks_and_bboxes(self, masks, indices, image_size):
    w, h = image_size

    # Create an empty canvas for the combined mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Initialize variables for bounding box combination
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Loop through the selected indices
    for idx in indices:
        # Combine the masks
        mask_array = masks[idx]["segmentation"].astype(np.uint8)
        combined_mask = np.logical_or(combined_mask, mask_array)

        # Update the bounding box
        x, y, w, h = masks[idx]["bbox"]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Convert the combined mask back to a PIL image
    combined_mask_image = Image.fromarray(combined_mask.astype(np.uint8) * 255)
    combined_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]  # Format as [x, y, width, height]

    return combined_mask_image, combined_bbox

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
