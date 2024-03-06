import numpy as np
import cv2
import os
from PIL import Image

# Specify the directory path
folder_path = 'Images 2/Segmentations'
# folder_path = 'Images/Segmentations'

# Get the list of files in the directory
files = os.listdir(folder_path)

# Sort the list of files based on the numbers they start with
sorted_files = sorted(files, key=lambda x: int(x.split('_')[0]) if '_' in x else float('inf'))
# file_names = [file for file in sorted_files]

# print(file_names)

# # Iterate through each file in the directory
# for filename in sorted_files:

#     img = cv2.imread(f'{folder_path}/{filename}', cv2.IMREAD_GRAYSCALE)
#     assert img is not None, "file could not be read, check with os.path.exists()"
#     img = cv2.medianBlur(img,5)
#     circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT_ALT,1,10,
#                                 param1=1,param2=0.85,minRadius=0,maxRadius=0)
    
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
        
#         for param in circles[0, :]:
        
#             print(filename)
#             print(f'OpenCV CoM: ({param[0]}, {param[1]})\nOpenCV Radius: {param[2]}')
#             print('-'*50)
#             cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#             # Draw the outer circle
#             cv2.circle(cimg,(param[0],param[1]),param[2],(0,255,0),2)
#             # Draw the center of the circle
#             cv2.circle(cimg,(param[0],param[1]),2,(0,0,255),3)
            
#             cv2.imshow('Detected Circles',cimg)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
            
#         continue

circle_detection_accuracy = 0.86

CV_deduced_CoM = []
CV_deduced_radius = []

# SAM_deduced_CoM = []
# SAM_deduced_radius = []

projection_idx = []

projection_num = '-'

for filename in sorted_files:

    if filename.split('_')[0] == projection_num and circle_found:
        continue
    elif filename.split('_')[0] != projection_num:
        projection_num = filename.split('_')[0]
        circle_found = False

    if circle_found:
        continue
    
    img = cv2.imread(f'{folder_path}/{filename}', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv2.medianBlur(img,5)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT_ALT,1,10,
                        param1=1,param2=circle_detection_accuracy,minRadius=0,maxRadius=0)
    
    if circles is not None:

        circles = np.uint16(np.around(circles))
        param = circles[0, 0]

        # bbox = mask['bbox']
        # radius = (bbox[2] + bbox[3]) / 4
        # radius = max(bbox[2], bbox[3]) / 2
        # CoM = [bbox[0] + radius, bbox[1] + radius]

        projection_idx.append(projection_num)

        CV_deduced_CoM.append([param[0], param[1]])
        CV_deduced_radius.append(param[2])

        # SAM_deduced_CoM.append(CoM)
        # SAM_deduced_radius.append(radius)

        circle_found = True

        print(f'Circle detected at projection number {projection_num}')

str_CV_xy_CoM = str(CV_deduced_CoM)

print(str_CV_xy_CoM)
print(projection_idx)
print(len(projection_idx))
print(CV_deduced_radius)

with open(f'/CV_xy_CoM.txt', 'w') as file:
    file.write(str_CV_xy_CoM)