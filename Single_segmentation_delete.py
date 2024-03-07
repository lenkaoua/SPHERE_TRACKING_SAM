from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import cv2
from PIL import Image

MODEL = 'vit_h'
CHECKPOINT = 'sam_vit_h_4b8939.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread('Pre-Segmentation.png')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
segmentation = mask_generator.generate(image_rgb)

for mask_num, mask in enumerate(segmentation):

    boolean_segmentation_array = mask['segmentation']
    integer_segmentation_array = boolean_segmentation_array.astype(int)
    
    image = Image.fromarray((integer_segmentation_array * 255).astype(np.uint8))
    image.save(f'Presentation/{mask_num}.png')