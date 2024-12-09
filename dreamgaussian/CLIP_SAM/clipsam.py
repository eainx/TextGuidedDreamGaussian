from .utils import segment_image, calculate_iou, image_resize, random_color
from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torch import nn
import numpy as np
import cv2
import torch


class ClipSeg:
    def __init__(self):
        # Load CLIPSeg
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    def segment(self, image, search_text: str) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(text=search_text, images=image, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = torch.sigmoid(outputs.logits)
        preds = nn.functional.interpolate(preds.unsqueeze(0),
                                          size=image.shape[:2],
                                          mode="bilinear").squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # cv2.imshow("preds", preds)

        return preds

class Sam:
    def __init__(self, sam, text):
        # Generate mask
        self.mask_generator = sam
        self.text = text
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clipseg = ClipSeg()

    def mask_image(self, path, W, H):
        # Open image and create mask
        img = cv2.imread(path)
        masks = self.mask_generator.generate(img)

        # Calculate clipseg score
        scores_clipseg = self.clipseg.segment(img, self.text)
        clipseg_mask = (scores_clipseg > 0.1).squeeze(2)  # numpy array
        clipseg_show = np.expand_dims(clipseg_mask, axis=-1).astype(np.uint8)
        # cv2.imshow("clipseg_mask", clipseg_show * 255)

        # Convert CLIPSeg mask to the same format as the image
        clipseg_mask = clipseg_mask.astype(np.uint8)

        # Original image for rgba
        alpha = np.ones((img.shape[0], img.shape[1], 1))
        img_rgba = np.concatenate((img, alpha), axis=-1)

        # Combined image
        combined_image = np.zeros(img_rgba.shape, dtype=img_rgba.dtype)

        # Overlay colored masks on the original image
        overlay_image = img.copy()

        for mask in masks:
            # Extract object based on SAM mask
            sam_mask = mask["segmentation"]
            segment = segment_image(img_rgba, sam_mask)

            # Calculate IoU with CLIPSeg mask
            iou = calculate_iou(sam_mask, clipseg_mask)

            # Combine masks based on IoU
            if iou > 0.5:  # Threshold
                # Mask for paste
                combined_image[sam_mask == 1] = segment[sam_mask == 1]

            # Create a colored mask with a random color
            color = random_color()
            overlay_image[sam_mask == 1] = color

        # Apply the colored mask on the original image
        overlay_image = cv2.addWeighted(img, 0.5, overlay_image, 0.5, 0)

        cv2.imshow("SAM RESULT", overlay_image)
        cv2.imwrite("./figure/overlay_cap.png", overlay_image)
        cv2.imwrite("./figure/cap.png", combined_image)

        # Convert color
        combined_image = combined_image[:, :, [2, 1, 0, 3]]

        final_image = image_resize(combined_image, W, H)

        return final_image
