from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from skimage import exposure
from tqdm import tqdm

def save_outputs(CV_xy_CoM, SAM_xy_CoM, CV_radii, SAM_radii, projection_idx, output_folder):

    str_CV_xy_CoM = str(CV_xy_CoM)
    str_SAM_xy_CoM = str(SAM_xy_CoM)
    str_CV_radii = str(CV_radii)
    str_SAM_radii = str(SAM_radii)
    str_projection_idx = str(projection_idx)
    
    with open(f'{output_folder}/CV_xy_CoM.txt', 'w') as file:
        file.write(str_CV_xy_CoM)
    with open(f'{output_folder}/SAM_xy_CoM.txt', 'w') as file:
        file.write(str_SAM_xy_CoM)
    with open(f'{output_folder}/CV_radii.txt', 'w') as file:
        file.write(str_CV_radii)
    with open(f'{output_folder}/SAM_radii.txt', 'w') as file:
        file.write(str_SAM_radii)
    with open(f'{output_folder}/projection_idx.txt', 'w') as file:
        file.write(str_projection_idx)

def circle_detection(segmentations, output_folder, circle_detection_accuracy=0.88):

    CV_deduced_CoM = []
    CV_deduced_radius = []

    SAM_deduced_CoM = []
    SAM_deduced_radius = []

    projection_idx = []

    for projection_num, segmentation in enumerate(segmentations):
    
        circle_found = False
        hollow_circle_found = False

        for mask_num, mask in enumerate(segmentation):

            if circle_found:
                continue
            
            boolean_segmentation_array = mask['segmentation']
            integer_segmentation_array = boolean_segmentation_array.astype(int)
            
            image = Image.fromarray((integer_segmentation_array * 255).astype(np.uint8))

            medianBlur_image = cv2.medianBlur(np.array(image),5)

            circles = cv2.HoughCircles(medianBlur_image,cv2.HOUGH_GRADIENT_ALT,1,10,
                                param1=1,param2=circle_detection_accuracy,minRadius=0,maxRadius=0)
            
            if circles is not None:
                
                circles = np.uint16(np.around(circles))
                param = circles[0, 0]

                bbox = mask['bbox']
                SAM_radius = max(bbox[2], bbox[3]) / 2
                SAM_CoM = [bbox[0] + SAM_radius, bbox[1] + SAM_radius]
                
                CV_radius = param[2]
                CV_CoM = [param[0], param[1]]

                if boolean_segmentation_array[CV_CoM[1], CV_CoM[0]]:
                    
                    projection_idx.append(projection_num)

                    CV_deduced_CoM.append(CV_CoM)
                    CV_deduced_radius.append(CV_radius)

                    SAM_deduced_CoM.append(SAM_CoM)
                    SAM_deduced_radius.append(SAM_radius)

                    circle_found = True

                    image.save(f'{output_folder}/Segmentations/{projection_num}_{mask_num}.png')

                    print(f'Circle detected at projection number {projection_num}')

                else:

                    hollow_circle_mask_image = image

                    hollow_circle_found = True

                    print(f'Hollow circle detected at projection number {projection_num}')
        
        if not circle_found and hollow_circle_found:

            projection_idx.append(projection_num)

            CV_deduced_CoM.append(CV_CoM)
            CV_deduced_radius.append(CV_radius)

            SAM_deduced_CoM.append(SAM_CoM)
            SAM_deduced_radius.append(SAM_radius)

            hollow_circle_mask_image.save(f'{output_folder}/Segmentations/{projection_num}_{mask_num}.png')

    return CV_deduced_CoM, CV_deduced_radius, SAM_deduced_CoM, SAM_deduced_radius, projection_idx

def segment_projections(projections):

    MODEL = 'vit_h'
    CHECKPOINT = 'sam_vit_h_4b8939.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device used for segmentation: {DEVICE}')

    segmentations = []

    sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT).to(device=DEVICE)
    
    mask_generator = SamAutomaticMaskGenerator(sam)

    for projection in tqdm(projections):
        
        image_object = Image.fromarray(projection)
        
        image = cv2.cvtColor(np.array(image_object), cv2.COLOR_BGR2RGB)

        segmentation = mask_generator.generate(image)

        segmentations.append(segmentation)

    return segmentations

def enhance_contrast(raw_projections):

    projections = []

    for raw_projection in raw_projections:

        # Apply histogram equalization for contrast enhancement
        projection = exposure.equalize_hist(raw_projection)

        # Rescale the image values to the range [0, 255]
        projection = (projection * 255).astype('uint8')
        
        projections.append(projection)

    return projections

def import_tiff_projections(file_path, NUMBER_OF_PROJECTIONS):

    all_projections = tifffile.imread(file_path)

    # Calculate the total number of images
    num_projections = len(all_projections)

    # Calculate the spacing between projections to select approximately 100 equally spaced images
    indices = np.linspace(0, num_projections - 1, NUMBER_OF_PROJECTIONS, dtype=int)
    
    images = all_projections[indices]

    print(f'Number of projections to process: {len(images)}')

    return images

def main():

    NUMBER_OF_PROJECTIONS = 1
    PIXEL_SIZE = 1.1e-6 # 1.1 μm
    ENERGY = 8e3 # 8 keV
    SOURCE_SAMPLE_DISTANCE = 220e-2 # 220 cm
    SAMPLE_DETECTOR_DISTANCE = 1e-2 # 1 cm
    SPHERE_RADIUS = 25e-6 # 40 μm
    SOURCE_DETECTOR_DISTANCE = SOURCE_SAMPLE_DISTANCE + SAMPLE_DETECTOR_DISTANCE # cm

    circle_detection_accuracy = 0.84

    file_path = 'TiffStack.tif'
    output_folder = 'Output'

    raw_projections = import_tiff_projections(file_path, NUMBER_OF_PROJECTIONS)
    projections = enhance_contrast(raw_projections)
    segmentations = segment_projections(projections)
    CV_xy_CoM, CV_radii, SAM_xy_CoM, SAM_radii, projection_idx = circle_detection(segmentations, output_folder, circle_detection_accuracy=circle_detection_accuracy)
    
    save_outputs(CV_xy_CoM, SAM_xy_CoM, CV_radii, SAM_radii, projection_idx, output_folder)

if __name__ == '__main__':
    main()