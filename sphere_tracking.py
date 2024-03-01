from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from skimage import exposure
import functools
import scipy.optimize
from tqdm import tqdm

def get_rotation_point(CoMs):

    rotation_point_list = []

    for CoM in CoMs:

        # Convert points to a NumPy array
        points_array = np.array(CoM)

        # Calculate the average of each dimension
        rotation_point = np.mean(points_array, axis=0)

        rotation_point_list.append(rotation_point)

    CV_rotation_point = rotation_point_list[0]
    SAM_rotation_point = rotation_point_list[1]

    print(f'\nCV point of rotation: {(CV_rotation_point[0], CV_rotation_point[1], CV_rotation_point[2])}, \nSAM point of rotation: {(SAM_rotation_point[0], SAM_rotation_point[1], SAM_rotation_point[2])}')

    return CV_rotation_point, SAM_rotation_point

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result

def get_rotation_axis(CoMs):

    rotation_axis_list = []

    for CoM in CoMs:

        fun = functools.partial(error, points=CoM)
        
        params0 = [0, 0, 0]
        res = scipy.optimize.minimize(fun, params0)

        a = res.x[0]
        b = res.x[1]

        A = [1,0,a]
        B = [0,1,b]

        normal = np.array([A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0]])

        # Swapping the order of the normal vector elements to adjust for the z-axis inversion
        normal = [normal[0],normal[2],normal[1]]

        # Normalising the normal vector to obtain the rotation axis
        rotation_axis = abs(normal / np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2))
        
        rotation_axis_list.append(rotation_axis)
    
    CV_rotation_axis = rotation_axis_list[0]
    SAM_rotation_axis = rotation_axis_list[1]

    print(f'\nCV axis of rotation: {(CV_rotation_axis[0], CV_rotation_axis[1], CV_rotation_axis[2])}, \nSAM axis of rotation: {(SAM_rotation_axis[0], SAM_rotation_axis[1], SAM_rotation_axis[2])}')

    return CV_rotation_axis, SAM_rotation_axis

def plot_sphere_trajectory(CV_CoM, SAM_CoM):

    # Extract x, y, z coordinates
    CV_x = [coord[0] for coord in CV_CoM]
    CV_y = [coord[1] for coord in CV_CoM]
    CV_z = [coord[2] for coord in CV_CoM]

    SAM_x = [coord[0] for coord in SAM_CoM]
    SAM_y = [coord[1] for coord in SAM_CoM]
    SAM_z = [coord[2] for coord in SAM_CoM]

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(CV_x, CV_y, CV_z, c='r', marker='o')
    ax.scatter(SAM_x, SAM_y, SAM_z, c='b', marker='o')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def deduce_z_axis_CoM(CV_CoM, CV_radii, SAM_CoM, SAM_radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE):

    # Convert the source to detector distance and sphere radius to pixel dimensions
    SOURCE_DETECTOR_DISTANCE = SOURCE_DETECTOR_DISTANCE / PIXEL_SIZE
    SPHERE_RADIUS = SPHERE_RADIUS / PIXEL_SIZE
    
    for i in range(len(CV_radii)):
        
        CV_magnification = SPHERE_RADIUS / CV_radii[i]
        CV_z_CoM = (SOURCE_DETECTOR_DISTANCE / CV_magnification) - SOURCE_DETECTOR_DISTANCE

        SAM_magnification = SPHERE_RADIUS / SAM_radii[i]
        SAM_z_CoM = (SOURCE_DETECTOR_DISTANCE / SAM_magnification) - SOURCE_DETECTOR_DISTANCE
        
        CV_CoM[i].append(CV_z_CoM)

        SAM_CoM[i].append(SAM_z_CoM)

    print(f'\nOpen CV CoMs: {CV_CoM}\nOpen CV Radii: {CV_radii}\nSAM CoMs: {SAM_CoM}\nSAM Radii: {SAM_radii}')

    return CV_CoM, SAM_CoM

def circle_detection(segmentations, NUMBER_OF_PROJECTIONS, output_folder):

    circle_detection_accuracy = 0.88

    CV_deduced_CoM = []
    CV_deduced_radius = []

    SAM_deduced_CoM = []
    SAM_deduced_radius = []

    projection_idx = []

    for projection_num, segmentation in enumerate(segmentations):
    
        circle_found = False

        for mask_num, mask in enumerate(segmentation):

            if circle_found:
                continue

            # print(f'PROJECTION NUM: {projection_num}')
            boolean_segmentation_array = mask['segmentation']
            integer_segmentation_array = boolean_segmentation_array.astype(int)
            
            image = Image.fromarray((integer_segmentation_array * 255).astype(np.uint8))
            # image.show()
            image.save(f'{output_folder}/Segmentations/{projection_num}_{mask_num}.png')

            image = cv2.medianBlur(np.array(image),5)

            circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT_ALT,1,10,
                                param1=1,param2=circle_detection_accuracy,minRadius=0,maxRadius=0)
            
            if circles is not None:
                
                circles = np.uint16(np.around(circles))
                param = circles[0, 0]

                circle_found = True

                if boolean_segmentation_array[param[1], param[0]]:
                    
                    bbox = mask['bbox']
                    # radius = (bbox[2] + bbox[3]) / 4
                    radius = max(bbox[2], bbox[3]) / 2
                    CoM = [bbox[0] + radius, bbox[1] + radius]

                    projection_idx.append(projection_num)

                    CV_deduced_CoM.append([param[0], param[1]])
                    CV_deduced_radius.append(param[2])

                    SAM_deduced_CoM.append(CoM)
                    SAM_deduced_radius.append(radius)

                    print(f'Circle detected at projection number {projection_num}')
                    # print(f'OpenCV CoM: ({param[0]}, {param[1]})\nOpenCV Radius: {param[2]}')
                    # print(f'SAM CoM: ({CoM[0]}, {CoM[1]})\nSAM Radius: {radius}')

                    # cimg = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

            #         # Draw the outer circle
            #         cv2.circle(cimg,(param[0],param[1]),param[2],(0,255,0),2)
            #         # Draw the center of the circle
            #         cv2.circle(cimg,(param[0],param[1]),2,(0,0,255),3)

            #         cv2.imshow('Detected Circles',cimg)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            #     else:
            #         print('Circle Outine Detected, Invalid.')
            # else:
            #     print('No circles detected.')

    return CV_deduced_CoM, CV_deduced_radius, SAM_deduced_CoM, SAM_deduced_radius, projection_idx

def segment_projections(projections):

    MODEL = 'vit_h'
    CHECKPOINT = 'sam_vit_h_4b8939.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(DEVICE)

    segmentations = []

    sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT).to(device=DEVICE)
    
    mask_generator = SamAutomaticMaskGenerator(sam)

    for projection in tqdm(projections):
        
        image_object = Image.fromarray(projection)
        
        image = cv2.cvtColor(np.array(image_object), cv2.COLOR_BGR2RGB)

        # Display the image using OpenCV
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        segmenatation = mask_generator.generate(image)

        segmentations.append(segmenatation)

    return segmentations

def enhance_contrast(raw_projections):

    projections = []

    for raw_projection in raw_projections:

        # Apply histogram equalization for contrast enhancement
        projection = exposure.equalize_hist(raw_projection)

        # Rescale the image values to the range [0, 255]
        projection = (projection * 255).astype('uint8')

        # plt.imshow(projection)
        # plt.axis('off')
        # plt.show()
        
        projections.append(projection)

    return projections


def import_tiff_projections(file_path, NUMBER_OF_PROJECTIONS):

    all_projections = tifffile.imread(file_path)

    # Calculate the total number of images
    number_of_projections = len(all_projections)

    # Calculate the spacing between projections to select approximately 100 equally spaced images
    projection_spacing = max(1, number_of_projections // NUMBER_OF_PROJECTIONS)
    
    images = all_projections[::projection_spacing]

    return images

def main():

    NUMBER_OF_PROJECTIONS = 1
    PIXEL_SIZE = 1.1e-6 # 1.1 μm
    ENERGY = 8e3 # 8 keV
    SOURCE_SAMPLE_DISTANCE = 220e-2 # 220 cm
    SAMPLE_DETECTOR_DISTANCE = 1e-2 # 1 cm
    SPHERE_RADIUS = 25e-6 # 40 μm
    SOURCE_DETECTOR_DISTANCE = SOURCE_SAMPLE_DISTANCE + SAMPLE_DETECTOR_DISTANCE # cm

    file_path = 'ProjectionsData.tiff'
    output_folder = 'Sphere Tracking Output'

    raw_projections = import_tiff_projections(file_path, NUMBER_OF_PROJECTIONS)
    projections = enhance_contrast(raw_projections)
    segmentations = segment_projections(projections)
    CV_xy_CoM, CV_radii, SAM_xy_CoM, SAM_radii, projection_idx = circle_detection(segmentations, NUMBER_OF_PROJECTIONS, output_folder)
    
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

    # CV_CoM, SAM_CoM = deduce_z_axis_CoM(CV_xy_CoM, CV_radii, SAM_xy_CoM, SAM_radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE)

    # plot_sphere_trajectory(CV_CoM, SAM_CoM)

    # # Get the rotation axis of the trajectory
    # CV_rotation_axis, SAM_rotation_axis = get_rotation_axis([CV_CoM, SAM_CoM])

    # # Get the point of rotation of the trajectory
    # CV_rotation_point, SAM_rotation_point =  get_rotation_point([CV_CoM, SAM_CoM])

if __name__ == '__main__':
    main()