import numpy as np
import cv2
import os

# Specify the directory path
folder_path = 'Output/Segmentations'
# folder_path = 'Images/Segmentations'

# Get the list of files in the directory
files = os.listdir(folder_path)

# Sort the list of files based on the numbers they start with
sorted_files = sorted(files, key=lambda x: int(x.split('_')[0]) if '_' in x else float('inf'))

# Iterate through each file in the directory
for filename in sorted_files:

    img = cv2.imread(f'{folder_path}/{filename}', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv2.medianBlur(img,5)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT_ALT,1,10,
                                param1=1,param2=0.88,minRadius=0,maxRadius=0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for param in circles[0, :]:
        
            print(filename)
            print(f'OpenCV CoM: ({param[0]}, {param[1]})\nOpenCV Radius: {param[2]}')
            print('-'*50)
            cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

            # Draw the outer circle
            cv2.circle(cimg,(param[0],param[1]),param[2],(0,255,0),2)
            # Draw the center of the circle
            cv2.circle(cimg,(param[0],param[1]),2,(0,0,255),3)
            
            cv2.imshow('Detected Circles',cimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        continue