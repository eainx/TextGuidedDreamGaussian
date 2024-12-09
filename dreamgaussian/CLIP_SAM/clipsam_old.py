from .utils import segment_image, calculate_iou, image_resize, random_color
from segment_anything import build_sam, SamAutomaticMaskGenerator
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
from torch import nn
import numpy as np
import cv2
import torch

class ClipSeg:
    def __init__(self):
        # Load CLIPSeg
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    def segment(self, image: Image.Image, search_text: str) -> int:
        image = image.convert('RGB')
        inputs = self.processor(text=search_text, images=image, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = torch.sigmoid(outputs.logits)
        preds = nn.functional.interpolate(preds.unsqueeze(0),
                                          size=(image.size[1], image.size[0]),
                                          mode="bilinear")
        preds = preds.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        cv2.imshow("preds", preds)

        return preds

class Sam:
    def __init__(self, text):
        # generate mask
        self.mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="/home/eainx/Documents/DL/FinalProject/dreamgaussian/CLIP_SAM/sam_vit_h_4b8939.pth"))
        self.text = text
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP
        # self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.clipseg = ClipSeg()

    # @torch.no_grad()
    # def retriev(self, elements: list[Image.Image], search_text: str) -> int:
    #     preprocessed_images = [self.preprocess(image).to('cuda') for image in elements]
    #     tokenized_text = clip.tokenize([search_text]).to('cuda')
    #     stacked_images = torch.stack(preprocessed_images)
    #     with torch.cuda.amp.autocast():
    #         image_features = self.model.encode_image(stacked_images)
    #         text_features = self.model.encode_text(tokenized_text)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #     probs = 100. * image_features @ text_features.T
    #     return probs[:, 0].softmax(dim=0)

    def mask_image(self, path, W, H):
        # Open image and create mask
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(img)

        # Open original image
        original_image = Image.open(path).convert('RGBA')

        # Calculate clipseg score
        scores_clipseg = self.clipseg.segment(original_image, self.text)
        clipseg_mask = (scores_clipseg > 0.1).squeeze(2)  # numpy array
        clipseg_show = np.expand_dims(clipseg_mask, axis=-1).astype(np.uint8)
        cv2.imshow("clipseg_mask", clipseg_show * 255.)

        # Convert CLIPSeg mask to the same format as the image
        clipseg_mask = Image.fromarray(clipseg_mask.astype(np.uint8))

        # Combined Image
        combined_image = Image.new('RGBA', original_image.size)

        # Overlay colored masks on the original image
        overlay_image = img.copy()

        for mask in masks:
            # Extract object based on SAM mask
            sam_mask = mask["segmentation"]
            segment = segment_image(original_image, sam_mask)

            # Calculate IoU with CLIPSeg mask
            iou = calculate_iou(sam_mask, clipseg_mask)

            # Combine masks based on IoU
            if iou > 0.5:  # Threshold
                # Mask for paste
                sam_mask_pil = Image.fromarray((sam_mask * 255).astype('uint8'), mode='L')
                # Paste
                combined_image.paste(segment, mask=sam_mask_pil)

                # Create a colored mask with a random color
                color = random_color()
                colored_mask = np.zeros_like(img)
                colored_mask[sam_mask == 1] = color

                # Apply the colored mask on the original image
                overlay_image = cv2.addWeighted(overlay_image, 1, colored_mask, 0.5, 0)

        combined_image.show()
        cv2.imshow("SAM RESULT", overlay_image)
        cv2.waitKey(0)

        final_image = image_resize(combined_image, W, H)


        return final_image


        # scores = self.retriev(cropped_boxes, self.text)


        # # seg_idx = get_indices_of_max_value(scores)[-1]
        # indices = get_indices_of_values_above_threshold(scores, 0.05)
        #
        # segmentation_masks = []
        # for seg_idx in indices:
        #     segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        #     segmentation_masks.append(segmentation_mask_image)
        #
        # combined_mask_image, combined_selected_box = self.combine_masks_and_bboxes(masks, indices, original_image.size)
        # combined_mask_image.show()
        #
        # # Create an empty mask for the entire image with transparency
        # mask_image = Image.new('L', original_image.size, 0)
        # draw = ImageDraw.Draw(mask_image)
        #
        # # segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        # # segmentation_masks.append(segmentation_mask_image)
        # # selected_box = masks[seg_idx]["bbox"]
        #
        #
        # draw.bitmap((0, 0), combined_mask_image, fill=255)
        # # mask_image.show()
        #
        # # Draw each segmentation mask onto the main mask image
        # # draw.bitmap((0, 0), segmentation_mask_image, fill=255)
        #
        # # Apply the inverted mask to the original image
        # cut_image = Image.composite(original_image, Image.new('RGBA', original_image.size, (0, 0, 0, 0)), mask_image)
        # cropped_cut_image = cut_image.crop(convert_box_xywh_to_xyxy(combined_selected_box))
        #
        #
        # # for seg_idx in indices:
        # #     segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        # #     segmentation_masks.append(segmentation_mask_image)
        # #
        # # overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        # # overlay_color = (255, 0, 0, 200)
        # #
        # # draw = ImageDraw.Draw(overlay_image)
        # #
        # # for segmentation_mask_image in segmentation_masks:
        # #     draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)
        # #
        # # result_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image, segmentation_mask_image)
        #
        # resized_cropped_cut_image = image_resize(cropped_cut_image, W, H)



