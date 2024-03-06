import numpy as np
import cv2
import os

def save_outputs(CoM, radii, projection_idx, output_folder):

    CoM = str(CoM)
    radii = str(radii)
    projection_idx = str(projection_idx)
    
    with open(f'{output_folder}/xy_CoM.txt', 'w') as file:
        file.write(CoM)
    with open(f'{output_folder}/radii.txt', 'w') as file:
        file.write(radii)
    with open(f'{output_folder}/projection_idx.txt', 'w') as file:
        file.write(projection_idx)

def circle_detection(segmentation_files, segmentation_files_path, circle_detection_accuracy, plot=False, disp=False):

    CoM = []
    radii = []
    projection_idx = []

    projection_num = None

    for filename in segmentation_files:

        if int(filename.split('_')[0]) == projection_num and circle_found:
            continue
        elif int(filename.split('_')[0]) != projection_num:
            projection_num = int(filename.split('_')[0])
            circle_found = False

        if circle_found:
            continue
        
        img = cv2.imread(f'{segmentation_files_path}/{filename}', cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        img = cv2.medianBlur(img,5)

        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT_ALT,1,10,
                            param1=1,param2=circle_detection_accuracy,minRadius=0,maxRadius=0)
        
        if circles is not None:

            circles = np.uint16(np.around(circles))
            param = circles[0, 0]

            projection_idx.append(projection_num)

            CoM.append([param[0], param[1]])
            radii.append(param[2])
            
            circle_found = True

            if plot:
                cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                cv2.circle(cimg,(param[0],param[1]),param[2],(0,255,0),2)
                # Draw the center of the circle
                cv2.circle(cimg,(param[0],param[1]),2,(0,0,255),3)

                cv2.imshow('Detected Circles',cimg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if disp:
                print(f'Circle detected at projection {projection_num}')
                print(f'CoM: {(param[0], param[1])}')
                print(f'Radii: {param[2]}')
                print('-'*50)

    return CoM, radii, projection_idx


def get_segmentation_files(segmentation_files_path):

    # Get the list of files in the directory
    segmentation_files = os.listdir(segmentation_files_path)

    # Sort the list of files based on the numbers they start with
    segmentation_files = sorted(segmentation_files, key=lambda x: int(x.split('_')[0]) if '_' in x else float('inf'))

    return segmentation_files

def main():

    # Specify the segmentation files path
    segmentation_files_path = 'Images 3/Segmentations'
    # Specify the output folder
    output_folder = 'Data Folder'
    # Set the circle detection tolerance-accuracy factor
    circle_detection_accuracy = 0.86

    segmentation_files = get_segmentation_files(segmentation_files_path)
    
    CoM, radii, projection_idx = circle_detection(segmentation_files, segmentation_files_path, circle_detection_accuracy, plot=False, disp=False)

    save_outputs(CoM, radii, projection_idx, output_folder)

if __name__ == '__main__':
    main()