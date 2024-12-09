from PhraseCutDataset.utils.refvg_loader import RefVGLoader
from PhraseCutDataset.utils.visualize_utils import plot_refvg_mask
from PhraseCutDataset.utils.data_transfer import polygons_to_mask
from dreamgaussian.CLIP_SAM.clipsam import Sam
from segment_anything import SamAutomaticMaskGenerator, build_sam, sam_model_registry
import cv2
import matplotlib
import time
from PIL import Image
import numpy as np

matplotlib.use('TkAgg')

class Evaluation:
    def __init__(self):
        self.refvg_loader = RefVGLoader(split='miniv')
        self.model_type = "vit_h"
        self.sam_base = sam_model_registry[self.model_type](checkpoint="/home/eainx/Documents/DL/FinalProject/dreamgaussian/CLIP_SAM/sam_vit_h_4b8939.pth")
        self.sam_base.to(device=self.device)
        # self.sam = SamAutomaticMaskGenerator(build_sam(checkpoint="/home/eainx/Documents/DL/FinalProject/dreamgaussian/CLIP_SAM/sam_vit_h_4b8939.pth"))
        self.sam = SamAutomaticMaskGenerator(self.sam_base)
    def load_image_as_mask(self, filepath):
        image = Image.open(filepath).convert("L")
        mask = np.array(image)
        mask = np.where(mask > 128, 1, 0)
        return mask

    # IoU 계산 함수
    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou


    def process_file(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        # 이미지와 해당 마스크 저장
        masks = {}
        updated_lines = []

        for line in lines:
            new_line = line.strip()
            parts = line.strip().split('\t')
            image_filename = parts[0]
            gt = self.load_image_as_mask('eval_gt/{}'.format(image_filename))
            ours = self.load_image_as_mask('eval_ours/{}'.format(image_filename))
            iou = self.calculate_iou(gt, ours)
            new_line += f"\tIoU: {iou:.4f}"
            updated_lines.append(new_line + '\n')

        # 파일에 변경사항을 다시 쓰기
        with open(filepath, 'w') as file:
            file.writelines(updated_lines)


    def get_ours(self):
        image_ids = self.refvg_loader.img_ids
        with open('image_phrase_old.txt', 'w') as file:
            for image_id in image_ids:
                img_ref_data = self.refvg_loader.get_img_ref_data(img_id=image_id)

                img_id = img_ref_data['image_id']

                for task_i, task_id in enumerate(img_ref_data['task_ids']):
                    phrase = img_ref_data['phrases'][task_i]
                    clipseg_sam = Sam(self.sam, phrase)
                    start_time = time.time()  # Start timing
                    image = clipseg_sam.mask_image('/media/eainx/2b2b0b48-d904-4abc-a9ce-60f34c87d9f2/PhraseCutDataset/data/VGPhraseCut_v0/images/{}.jpg'.format(img_id))
                    elapsed_time = time.time() - start_time   # End timing
                    cv2.imwrite("eval_ours/{}_{}.png".format(img_id, task_i), image)
                    # Write image path and phrase to file
                    file.write("{}_{}.png\t{}\t{:.2f}seconds\n".format(img_id, task_i, phrase, elapsed_time))
                    file.flush()
    def get_gt(self):
        self.refvg_loader = RefVGLoader(split='miniv')

        image_ids = self.refvg_loader.img_ids

        for image_id in image_ids:
            img_ref_data = self.refvg_loader.get_img_ref_data(img_id=image_id)

            img_id = img_ref_data['image_id']
            height, width = img_ref_data['height'], img_ref_data['width']

            for task_i, task_id in enumerate(img_ref_data['task_ids']):
                gt_Polygons = img_ref_data['gt_Polygons'][task_i]
                for ins_i, ins_ps in enumerate(gt_Polygons):
                    mps = polygons_to_mask(ins_ps, width, height)
                gt_masks = mps
                fig = plot_refvg_mask(img_id=img_id, gray_img=False, gt_mask=gt_masks)
                fig.save("eval/{}_{}.png".format(img_id, task_i))


if __name__ == "__main__":
    eval = Evaluation()
    eval.get_ours()