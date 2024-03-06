import numpy as np
import matplotlib.pyplot as plt
import functools
import scipy.optimize
import tifffile
import ast
from skimage.transform import iradon
from scipy.optimize import curve_fit

def correct_data(poly, CV_CoM, images, NUMBER_OF_PROJECTIONS, background_value):
    corrected_images = []

    count = 0

    for image, (x, y, z) in zip(images, CV_CoM):
        expected_y = poly(count)
        count += 1

        shift = expected_y - y  # Calculate vertical shift based on the difference
        # Creating a new array filled with NaNs
        corrected_image = np.full_like(image, background_value, dtype=np.float64)
        
        if shift > 0:  # Invert: Shift the projection up
            corrected_image[:int(-shift) or None, :] = image[int(shift):, :]
        else:  # Invert: Shift the projection down
            corrected_image[int(-shift):, :] = image[:int(shift) or None, :]

        corrected_images.append(corrected_image)
    
    return corrected_images

"""
def correct_data(poly, CV_CoM, projections, NUMBER_OF_PROJECTIONS):

    CV_x_components = [point[1] for point in CV_CoM]

    # Calculate the residuals for CV and SAM curves
    CV_x_residuals = CV_x_components - poly(list(range(NUMBER_OF_PROJECTIONS)))
    # Calculate the maximum and minimum deltas across all projections
    max_delta = max(abs(CV_x_residuals))
    min_delta = min(abs(CV_x_residuals))

    # Calculate the maximum padding needed for any projection
    max_padding = max(max_delta, min_delta)

    padded_projections = []

    for i, projection in enumerate(projections):
        delta_CV_x = CV_x_components[i] - poly(i)
        delta_CV_x = int(delta_CV_x)

        # Determine how many NaN rows to add to the top and bottom of the projection
        if delta_CV_x > 0:
            extra_top_rows = np.full((delta_CV_x, projection.shape[1]), np.nan)
            extra_bottom_rows = np.zeros((0, projection.shape[1]))  # No NaNs needed at the bottom
        elif delta_CV_x < 0:
            extra_top_rows = np.zeros((0, projection.shape[1]))  # No NaNs needed at the top
            extra_bottom_rows = np.full((abs(delta_CV_x), projection.shape[1]), np.nan)
        else:
            extra_top_rows = np.zeros((0, projection.shape[1]))  # No NaNs needed
            extra_bottom_rows = np.zeros((0, projection.shape[1]))  # No NaNs needed

        # Stack the extra rows with the projection, ensuring homogeneity
        padded_projection = np.vstack([extra_bottom_rows[:min_delta], projection, extra_top_rows[:max_delta]])
        padded_projections.append(padded_projection)

    # Convert the list of arrays to a single numpy array
    padded_projections = np.array(padded_projections)
"""
"""
def correct_data(poly, CV_CoM, projections, NUMBER_OF_PROJECTIONS):

    CV_x_components = [point[1] for point in CV_CoM]

    # Calculate the residuals for CV and SAM curves
    CV_x_residuals = CV_x_components - poly(list(range(NUMBER_OF_PROJECTIONS)))

    # Find the index of the point with the maximum residual for both CV and SAM
    max_CV_x_residual_idx = int(abs(max(CV_x_residuals)))
    min_CV_x_residual_idx = int(abs(min(CV_x_residuals)))

    padded_projections = []  # Initialize an empty list

    for i, projection in enumerate(projections):
        
        extra_top_rows = np.full((max_CV_x_residual_idx, projection.shape[1]), np.nan)
        extra_bottom_rows = np.full((min_CV_x_residual_idx, projection.shape[1]), np.nan)
        padded_projection = np.vstack([extra_bottom_rows, projection, extra_top_rows])

        padded_projections.append(padded_projection)  # Append the padded projection to the list
    
    # Convert the list of arrays to a single numpy array
    padded_projections = np.array(padded_projections)

    # corrected_projections = []

    corrected_projections = []

    for i, projection in enumerate(padded_projections):
        print(CV_x_components[i])
        print(poly(i))
        delta_CV_x = CV_x_components[i] - poly(i)
        delta_CV_x = int(delta_CV_x)

        print(int(delta_CV_x))

        if delta_CV_x > 0:
            
            trimmed_projection = projection[delta_CV_x:]
            extra_top_rows = np.full((abs(delta_CV_x), trimmed_projection.shape[1]), np.nan)
            corrected_projection = np.vstack([trimmed_projection, extra_top_rows])

            corrected_projections.append(corrected_projection)
        
        elif delta_CV_x < 0:
            
            trimmed_projection = projection[:delta_CV_x]
            extra_bottom_rows = np.full((abs(delta_CV_x), trimmed_projection.shape[1]), np.nan)
            corrected_projection = np.vstack([extra_bottom_rows, trimmed_projection])

            corrected_projections.append(corrected_projection)

        else:

            corrected_projections.append(projection)

    corrected_projections = np.array(corrected_projections)

    return corrected_projections
"""

def iradon_reconstruction(CV_sinogram_array, SAM_sinogram_array, background_value, shift):

    number_of_projections = CV_sinogram_array.shape[1]
    print(number_of_projections)

    theta = np.linspace(0., 360., int(number_of_projections), endpoint=False)

    print("Length of theta array:", len(theta))
    print("Shape of CV_sinogram_array:", CV_sinogram_array.shape)

    CV_sinogram_array = -np.log(CV_sinogram_array/background_value)
    CV_sinogram_array = np.roll(CV_sinogram_array, shift, axis=0)

    SAM_sinogram_array = -np.log(SAM_sinogram_array/background_value)
    SAM_sinogram_array = np.roll(SAM_sinogram_array, shift, axis=0)

    CV_reconstruction_fbp = iradon(CV_sinogram_array, theta=theta, filter_name='ramp')
    SAM_reconstruction_fbp = iradon(SAM_sinogram_array, theta=theta, filter_name='ramp')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                sharex=True, sharey=True)
    ax1.set_title('CV Filtered back projection reconstruction')
    ax1.imshow(CV_reconstruction_fbp, cmap=plt.cm.Greys_r)

    ax2.set_title('SAM Filtered back projection reconstruction')
    ax2.imshow(SAM_reconstruction_fbp, cmap=plt.cm.Greys_r)

    plt.show()

def add_missing_projections(CV_x_poly, SAM_x_poly, CV_y_poly, SAM_y_poly, CV_CoM, SAM_CoM, projection_idx, NUMBER_OF_PROJECTIONS):

    for i in range(NUMBER_OF_PROJECTIONS):

        if i not in projection_idx:
            CV_x = CV_x_poly(i)
            CV_y = CV_y_poly(i)
            CV_z = np.nan

            SAM_x = SAM_x_poly(i)
            SAM_y = SAM_y_poly(i)
            SAM_z = np.nan

            CV_CoM.insert(i, [CV_x, CV_y, CV_z])
            SAM_CoM.insert(i, [SAM_x, SAM_y, SAM_z])

    return CV_CoM, SAM_CoM

def y_sinograms(CoMs, projections, projection_idx, NUMBER_OF_PROJECTIONS, complete=False):

    y_sinograms = []

    for CoM_num, CoM in enumerate(CoMs):

        y_sinogram = []
        
        if complete:

            for i in range(NUMBER_OF_PROJECTIONS):

                projection = projections[CoM_num][i]

                y_CoM = int(CoM[i][0])
                x = projection[:,y_CoM]
                y_sinogram.append(x)
            
        else:

            for i, idx in enumerate(projection_idx):

                projection = projections[CoM_num][idx]

                y_CoM = int(CoM[i][0])
                x = projection[:,y_CoM]
                y_sinogram.append(x)

        y_sinogram = np.transpose(y_sinogram)

        y_sinograms.append(y_sinogram)

    CV_y_sinogram = y_sinograms[0]
    SAM_y_sinogram = y_sinograms[1]

    return CV_y_sinogram, SAM_y_sinogram

def x_sinograms(CoMs, projections, projection_idx, NUMBER_OF_PROJECTIONS, complete=False):

    x_sinograms = []

    for CoM_num, CoM in enumerate(CoMs):

        x_sinogram = []
        
        if complete:

            for i in range(NUMBER_OF_PROJECTIONS):

                projection = projections[CoM_num][i]

                x_CoM = int(CoM[i][1])
                x = projection[x_CoM,:]
                x_sinogram.append(x)
            
        else:

            for i, idx in enumerate(projection_idx):

                projection = projections[CoM_num][idx]

                x_CoM = int(CoM[i][1])
                x = projection[x_CoM,:]
                x_sinogram.append(x)

        x_sinogram = np.transpose(x_sinogram)

        x_sinograms.append(x_sinogram)

    CV_x_sinogram = x_sinograms[0]
    SAM_x_sinogram = x_sinograms[1]

    return CV_x_sinogram, SAM_x_sinogram

def plot_sinograms(CV_sinogram, SAM_sinogram):

    # Create subplots
    fig, (ax1, ax2)  = plt.subplots(1, 2)

    # Plot CV_sinogram
    ax1.imshow(CV_sinogram, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.set_title('CV Sinogram')

    # Plot SAM_sinogram
    ax2.imshow(SAM_sinogram, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title('SAM Sinogram')

    plt.tight_layout()
    plt.show()

def y_curve_fitting(CV_CoM, SAM_CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=False):

    CV_y_CoM = [y[1] for y in CV_CoM]
    SAM_y_CoM = [y[1] for y in SAM_CoM]

    # Fit a 3rd order polynomial for CV_y_CoM
    CV_coeff = np.polyfit(projection_idx, CV_y_CoM, 12)
    CV_poly = np.poly1d(CV_coeff)

    # Fit a 3rd order polynomial for SAM_y_CoM
    SAM_coeff = np.polyfit(projection_idx, SAM_y_CoM, 12)
    SAM_poly = np.poly1d(SAM_coeff)

    # Generate points for the fitted polynomials
    x_fit = np.linspace(0, NUMBER_OF_PROJECTIONS, 100)
    CV_y_fit = CV_poly(x_fit)
    SAM_fit = SAM_poly(x_fit)
    

    if plot:

        # Create figure and axes objects with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

        # Plot for the first subplot (CV)
        ax1.scatter(projection_idx, CV_y_CoM, marker='o', color='red', label='Data', s=3)
        ax1.plot(x_fit, CV_y_fit, 'k-', label='Fitted curve (CV)', linewidth=1)
        ax1.set_xlabel('Projection Number')
        ax1.set_ylabel('CV Y-axis')
        ax1.set_title('CV Sinogram along Y-axis')
        ax1.legend()
        ax1.grid(True)

        # Plot for the second subplot (SAM)
        ax2.scatter(projection_idx, SAM_y_CoM, marker='o', color='blue', label='Data', s=3)
        ax2.plot(x_fit, SAM_fit, 'k-', label='Fitted curve (SAM)', linewidth=1)
        ax2.set_xlabel('Projection Number')
        ax2.set_ylabel('SAM Y-axis')
        ax2.set_title('SAM Sinogram along Y-axis')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

    return CV_poly, SAM_poly


def x_curve_fitting(CV_CoM, SAM_CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=False):
    
    CV_x_CoM = [x[0] for x in CV_CoM]
    SAM_x_CoM = [x[0] for x in SAM_CoM]

    # Fit a 1st degree polynomial for CV_y_CoM (linear regression)
    CV_coeff = np.polyfit(projection_idx, CV_x_CoM, 1)
    CV_poly = np.poly1d(CV_coeff)

    # Fit a 1st degree polynomial for SAM_y_CoM (linear regression)
    SAM_coeff = np.polyfit(projection_idx, SAM_x_CoM, 1)
    SAM_poly = np.poly1d(SAM_coeff)

    # Generate points for the fitted polynomials
    x_fit = np.linspace(0, NUMBER_OF_PROJECTIONS, 100)
    CV_y_fit = CV_poly(x_fit)
    SAM_y_fit = SAM_poly(x_fit)

    if plot:

        # Create figure and axes objects with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

        # Plot for the first subplot (CV)
        ax1.scatter(projection_idx, CV_x_CoM, marker='o', color='red', label='Data', s=10)
        ax1.plot(x_fit, CV_y_fit, 'k-', label='Fitted line (CV)')
        ax1.set_xlabel('Projection Number')
        ax1.set_ylabel('CV X-axis')
        ax1.set_title('CV Sinogram along X-axis')
        ax1.legend()
        ax1.grid(True)

        # Plot for the second subplot (SAM)
        ax2.scatter(projection_idx, SAM_x_CoM, marker='o', color='blue', label='Data', s=10)
        ax2.plot(x_fit, SAM_y_fit, 'k-', label='Fitted line (SAM)')
        ax2.set_xlabel('Projection Number')
        ax2.set_ylabel('SAM X-axis')
        ax2.set_title('SAM Sinogram along X-axis')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

    return CV_poly, SAM_poly

def plot_trajectory(CV_CoM, SAM_CoM, CV_rotation_axis, SAM_rotation_axis, CV_rotation_point, SAM_rotation_point):

    CV_x_CoM = [x[0] for x in CV_CoM]
    CV_y_CoM = [y[1] for y in CV_CoM]
    CV_z_CoM = [z[2] for z in CV_CoM]

    SAM_x_CoM = [x[0] for x in SAM_CoM]
    SAM_y_CoM = [y[1] for y in SAM_CoM]
    SAM_z_CoM = [z[2] for z in SAM_CoM]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the deduced positions of the sphere in red
    ax.scatter(CV_x_CoM, CV_y_CoM, CV_z_CoM, c='r', marker='o', label='CV CoM')

    # Plot the deduced point of rotation
    ax.scatter(CV_rotation_point[0], CV_rotation_point[1], CV_rotation_point[2], c='r', marker='x', label='CV Point of Rotation')

    # Plot the deduced rotation axis
    ax.quiver(CV_rotation_point[0], CV_rotation_point[1], CV_rotation_point[2], CV_rotation_axis[0], CV_rotation_axis[2], CV_rotation_axis[1], length=50, color='r', label='CV Axis of Rotation')

    # Plot the deduced positions of the sphere in red
    ax.scatter(SAM_x_CoM, SAM_y_CoM, SAM_z_CoM, c='b', marker='o', label='SAM CoM')

    # Plot the deduced point of rotation
    ax.scatter(SAM_rotation_point[0], SAM_rotation_point[1], SAM_rotation_point[2], c='b', marker='x', label='SAM Point of Rotation')

    # Plot the deduced rotation axis
    ax.quiver(SAM_rotation_point[0], SAM_rotation_point[1], SAM_rotation_point[2], SAM_rotation_axis[0], SAM_rotation_axis[2], SAM_rotation_axis[1], length=50, color='b', label='SAM Axis of Rotation')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory of the Sphere')

    # Rotate the view
    ax.view_init(elev=25, azim=45, vertical_axis='y')

    # Get the current axes
    ax = plt.gca()
    
    # # # Reset the axes limits to their default values
    # ax.set_xlim(65, 85)
    # # ax.set_ylim(-1000, 2000)
    # ax.set_zlim(-0000, -300000)

    ax.legend()
    plt.show()
    plt.close(fig)

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

def deduce_z_axis_CoM(CV_CoM, CV_radii, SAM_CoM, SAM_radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE):

    # Convert the source to detector distance and sphere radius to pixel dimensions
    SOURCE_DETECTOR_DISTANCE = SOURCE_DETECTOR_DISTANCE / PIXEL_SIZE
    SPHERE_RADIUS = SPHERE_RADIUS / PIXEL_SIZE
    
    for i in range(len(CV_CoM)):
        
        CV_magnification = SPHERE_RADIUS / CV_radii[i]
        CV_z_CoM = (SOURCE_DETECTOR_DISTANCE / CV_magnification) - SOURCE_DETECTOR_DISTANCE

        SAM_magnification = SPHERE_RADIUS / SAM_radii[i]
        SAM_z_CoM = (SOURCE_DETECTOR_DISTANCE / SAM_magnification) - SOURCE_DETECTOR_DISTANCE
        
        CV_CoM[i].append(CV_z_CoM)

        SAM_CoM[i].append(SAM_z_CoM)

    # print(f'\nOpen CV CoMs: {CV_CoM}\nOpen CV Radii: {CV_radii}\nSAM CoMs: {SAM_CoM}\nSAM Radii: {SAM_radii}')

    return CV_CoM, SAM_CoM

def import_tiff_projections(file_path, NUMBER_OF_PROJECTIONS):

    all_projections = tifffile.imread(file_path)

    # Calculate the total number of images
    num_projections = len(all_projections)

    # Calculate the spacing between projections to select approximately 100 equally spaced images
    indices = np.linspace(0, num_projections - 1, NUMBER_OF_PROJECTIONS, dtype=int)
    
    images = all_projections[indices]

    first_image = images[0]

    # Get the dimensions of the first image
    image_height, image_width = first_image.shape

    return images, image_height, image_width

def plot_raw_sinogram(projections, NUMBER_OF_PROJECTIONS, shift_up, background_value):

    sinogram = []

    for i in range(NUMBER_OF_PROJECTIONS):

        projection = projections[i]

        x = projection[:,20]
        sinogram.append(x)

    sinogram = np.transpose(sinogram)

    sinogram = np.roll(sinogram, shift_up, axis=0)
    
    # Create subplots
    fig, (ax1, ax2)  = plt.subplots(1, 2)

    theta = np.linspace(0., 360., 652, endpoint=False)

    sinogram = -np.log(sinogram/background_value)

    CV_reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')

    ax1.set_title('Reconstruction')

    im1 = ax1.imshow(sinogram, aspect='auto', cmap='viridis')
    im2 = ax2.imshow(CV_reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.set_title('1st Column Sinogram')
    fig.colorbar(im1)
    fig.colorbar(im2)
    plt.tight_layout()
    plt.show()

    return sinogram

def determine_roll(CV_y_poly, NUMBER_OF_PROJECTIONS, image_height, image_width):

    min_x = CV_y_poly(0)
    max_x = CV_y_poly(NUMBER_OF_PROJECTIONS)

    print(min_x)
    print(max_x)

    middle = image_height / 2

    actual_middle = (max_x - min_x) / 2 + min_x

    shift = (middle - actual_middle) / 2

    print(middle, actual_middle)

    return 0 

def determine_background_value(CV_x_sinogram_array):

    background_value = np.average(CV_x_sinogram_array[:,2])

    print(background_value)

    return background_value


def main():

    PIXEL_SIZE = 1.1e-6 # 1.1 μm
    SOURCE_SAMPLE_DISTANCE = 220e-2 # 220 cm
    SAMPLE_DETECTOR_DISTANCE = 1e-2 # 1 cm
    SPHERE_RADIUS = 25e-6 # 40 μm
    SOURCE_DETECTOR_DISTANCE = SOURCE_SAMPLE_DISTANCE + SAMPLE_DETECTOR_DISTANCE # cm
    NUMBER_OF_PROJECTIONS = 652

    projections_file_path = 'TiffStack.tif'
    data_folder = 'Images 2'

    # Open the text file containing the array
    with open(f'{data_folder}/CV_xy_CoM.txt', 'r') as file:
        # Read the contents of the file
        data = file.read()
        # Use ast.literal_eval() to convert the string to a Python object (list of lists)
        CV_xy_CoM = ast.literal_eval(data)

    # Open the text file containing the array
    with open(f'{data_folder}/SAM_xy_CoM.txt', 'r') as file:
        # Read the contents of the file
        data = file.read()
        # Use ast.literal_eval() to convert the string to a Python object (list of lists)
        SAM_xy_CoM = ast.literal_eval(data)
    
        # Open the text file containing the array
    with open(f'{data_folder}/CV_radii.txt', 'r') as file:
        # Read the contents of the file
        data = file.read()
        # Use ast.literal_eval() to convert the string to a Python object (list of lists)
        CV_radii = ast.literal_eval(data)

    with open(f'{data_folder}/SAM_radii.txt', 'r') as file:
        # Read the contents of the file
        data = file.read()
        # Use ast.literal_eval() to convert the string to a Python object (list of lists)
        SAM_radii = ast.literal_eval(data)

    with open(f'{data_folder}/projection_idx.txt', 'r') as file:
        # Read the contents of the file
        data = file.read()
        # Use ast.literal_eval() to convert the string to a Python object (list of lists)
        projection_idx = ast.literal_eval(data)

    CV_xy_CoM = [[582, 32], [578, 32], [582, 32], [577, 32], [578, 32], [583, 33], [579, 32], [579, 32], [576, 32], [575, 32], [578, 31], [574, 33], [575, 32], [579, 33], [574, 32], [575, 32], [577, 32], [582, 31], [575, 32], [573, 31], [575, 32], [579, 33], [574, 32], [572, 32], [569, 32], [572, 32], [571, 32], [572, 32], [566, 33], [572, 32], [562, 31], [564, 32], [563, 30], [566, 31], [559, 32], [564, 32], [565, 32], [557, 32], [556, 31], [560, 31], [561, 32], [559, 32], [550, 32], [560, 32], [559, 32], [550, 32], [556, 31], [550, 32], [550, 31], [553, 32], [548, 33], [550, 32], [546, 33], [541, 32], [543, 32], [543, 32], [539, 32], [538, 32], [542, 32], [534, 32], [535, 32], [534, 32], [534, 32], [528, 31], [526, 32], [530, 32], [529, 32], [530, 32], [521, 32], [520, 33], [522, 32], [518, 31], [520, 31], [517, 32], [511, 32], [513, 32], [515, 32], [507, 31], [506, 32], [505, 32], [503, 32], [500, 32], [503, 32], [504, 32], [502, 32], [499, 31], [497, 32], [494, 31], [491, 32], [487, 32], [487, 31], [488, 32], [481, 32], [487, 32], [483, 32], [478, 33], [475, 32], [474, 32], [472, 32], [468, 32], [462, 32], [457, 32], [464, 32], [453, 32], [455, 31], [447, 31], [448, 31], [442, 32], [442, 32], [433, 32], [440, 30], [435, 32], [435, 32], [433, 31], [430, 31], [427, 31], [418, 31], [417, 31], [413, 32], [418, 32], [412, 32], [408, 32], [404, 32], [398, 32], [401, 31], [388, 32], [388, 31], [383, 32], [383, 32], [381, 32], [374, 32], [376, 32], [378, 32], [375, 31], [364, 33], [366, 32], [359, 32], [365, 33], [354, 32], [357, 31], [349, 32], [352, 32], [350, 32], [348, 32], [341, 32], [344, 32], [342, 31], [333, 33], [334, 32], [330, 32], [326, 32], [320, 32], [327, 32], [320, 32], [322, 32], [317, 32], [316, 32], [308, 33], [310, 32], [302, 32], [302, 32], [295, 32], [296, 31], [291, 32], [289, 32], [284, 31], [286, 30], [283, 32], [279, 32], [277, 33], [278, 32], [273, 31], [272, 32], [269, 32], [264, 31], [261, 32], [264, 32], [262, 33], [253, 32], [251, 32], [252, 31], [249, 32], [254, 32], [246, 32], [247, 32], [240, 32], [239, 32], [241, 32], [242, 32], [236, 32], [238, 32], [229, 32], [230, 32], [227, 32], [226, 32], [225, 32], [221, 31], [223, 31], [221, 32], [220, 32], [217, 32], [210, 31], [207, 32], [214, 32], [206, 31], [207, 32], [206, 32], [207, 32], [196, 32], [200, 31], [200, 33], [199, 31], [198, 32], [195, 32], [190, 32], [188, 32], [185, 32], [187, 33], [190, 33], [187, 33], [179, 33], [175, 32], [177, 32], [177, 32], [171, 32], [172, 32], [176, 31], [168, 32], [172, 32], [169, 33], [169, 32], [159, 32], [158, 32], [159, 32], [161, 32], [157, 32], [156, 32], [156, 32], [151, 32], [152, 32], [148, 31], [143, 32], [147, 31], [149, 32], [150, 32], [142, 31], [141, 32], [137, 32], [143, 32], [140, 32], [140, 31], [136, 31], [135, 32], [142, 32], [135, 32], [134, 32], [130, 32], [134, 32], [128, 31], [136, 32], [127, 33], [132, 33], [125, 31], [131, 32], [127, 31], [125, 32], [129, 32], [128, 32], [126, 31], [122, 33], [129, 32], [123, 32], [123, 32], [120, 31], [124, 31], [119, 31], [121, 32], [125, 31], [124, 33], [118, 32], [118, 32], [115, 32], [119, 32], [116, 31], [119, 32], [118, 31], [118, 31], [117, 33], [123, 32], [118, 32], [124, 32], [122, 32], [124, 33], [124, 31], [124, 31], [123, 32], [120, 32], [122, 32], [121, 33], [123, 32], [119, 32], [120, 32], [126, 31], [126, 32], [120, 32], [128, 32], [121, 32], [129, 31], [126, 32], [129, 32], [125, 32], [125, 32], [132, 32], [129, 32], [128, 32], [130, 32], [135, 32], [136, 32], [133, 32], [131, 32], [129, 32], [138, 32], [140, 32], [132, 33], [133, 31], [138, 32], [137, 32], [134, 32], [139, 32], [140, 33], [141, 32], [145, 31], [147, 32], [150, 30], [148, 32], [149, 32], [148, 32], [156, 31], [150, 32], [160, 32], [152, 32], [154, 31], [162, 31], [162, 32], [161, 31], [167, 32], [166, 32], [168, 32], [166, 32], [166, 32], [165, 33], [170, 32], [170, 32], [173, 32], [176, 32], [174, 31], [180, 30], [184, 31], [179, 32], [179, 32], [190, 33], [184, 32], [184, 32], [189, 32], [193, 33], [188, 32], [197, 32], [193, 31], [202, 32], [202, 33], [198, 33], [204, 32], [206, 31], [205, 32], [206, 32], [210, 31], [215, 32], [216, 32], [219, 32], [221, 32], [223, 32], [222, 32], [230, 32], [230, 32], [230, 32], [230, 32], [241, 32], [237, 32], [240, 32], [244, 32], [249, 32], [249, 32], [259, 32], [261, 32], [259, 31], [264, 31], [267, 32], [271, 32], [276, 32], [275, 31], [272, 32], [280, 32], [277, 32], [281, 32], [283, 32], [286, 31], [290, 31], [290, 32], [297, 33], [294, 31], [302, 31], [298, 30], [304, 31], [314, 33], [322, 31], [325, 32], [327, 32], [325, 32], [332, 32], [335, 32], [332, 32], [339, 31], [337, 31], [343, 31], [338, 32], [348, 33], [349, 32], [345, 32], [348, 32], [349, 32], [351, 32], [361, 32], [356, 32], [358, 31], [366, 32], [362, 32], [369, 32], [375, 32], [371, 33], [377, 32], [382, 32], [378, 32], [387, 32], [392, 32], [393, 31], [391, 30], [396, 32], [400, 32], [401, 31], [408, 32], [402, 32], [405, 32], [406, 32], [410, 33], [415, 32], [420, 32], [419, 32], [423, 31], [425, 32], [422, 32], [430, 32], [434, 32], [425, 31], [432, 32], [436, 32], [441, 32], [440, 32], [438, 31], [446, 32], [443, 31], [451, 31], [444, 32], [448, 31], [457, 32], [459, 31], [455, 31], [462, 32], [459, 32], [462, 32], [469, 32], [469, 32], [468, 32], [469, 32], [474, 32], [476, 32], [479, 31], [474, 32], [482, 32], [480, 31], [480, 32], [485, 32], [485, 32], [493, 32], [492, 33], [496, 31], [499, 32], [499, 32], [493, 31], [497, 33], [500, 32], [498, 32], [508, 31], [511, 32], [506, 32], [507, 31], [514, 32], [514, 32], [517, 32], [514, 32], [517, 33], [515, 32], [518, 32], [519, 31], [528, 32], [530, 32], [523, 32], [525, 32], [526, 32], [530, 32], [532, 31], [532, 31], [537, 32], [540, 32], [539, 32], [541, 31], [544, 32], [536, 32], [547, 31], [304, 55], [547, 32], [545, 32], [545, 32], [547, 32], [554, 31], [550, 33], [547, 31], [547, 31], [553, 32], [559, 31], [556, 32], [558, 31], [555, 32], [562, 31], [565, 32], [563, 31], [567, 32], [564, 31], [561, 32], [564, 32], [562, 32], [564, 31], [566, 31], [572, 32], [565, 32], [572, 32], [576, 31], [575, 32], [574, 32], [571, 32], [578, 33], [571, 31], [578, 33], [576, 32], [576, 32], [574, 32], [575, 31], [579, 32], [581, 32], [581, 32], [580, 31], [575, 32], [577, 32], [577, 31], [576, 32], [577, 32], [586, 32], [576, 31], [584, 32], [585, 32], [579, 32], [583, 32]]

    SAM_xy_CoM = CV_xy_CoM

    projection_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 109, 111, 112, 113, 115, 116, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 140, 141, 144, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 183, 184, 186, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255, 256, 257, 258, 260, 261, 263, 264, 265, 266, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 301, 302, 304, 305, 306, 307, 309, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 347, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 376, 377, 378, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 418, 419, 420, 421, 422, 423, 424, 426, 427, 428, 429, 430, 431, 433, 434, 436, 437, 439, 440, 442, 443, 445, 446, 448, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 463, 464, 465, 466, 467, 469, 471, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 504, 506, 507, 508, 509, 510, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 608, 609, 610, 611, 612, 614, 615, 616, 617, 619, 621, 622, 623, 624, 625, 626, 628, 629, 630, 632, 633, 634, 635, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651]

    CV_xy_CoM = [[sublist[1], sublist[0]] for sublist in CV_xy_CoM]
    SAM_xy_CoM = [[sublist[1], sublist[0]] for sublist in SAM_xy_CoM]

    CV_radii = [9, 9, 10, 9, 9, 9, 9, 9, 10, 10, 9, 10, 9, 9, 9, 10, 9, 9, 10, 9, 10, 9, 10, 9, 10, 10, 10, 9, 10, 10, 9, 9, 10, 10, 9, 9, 9, 9, 10, 10, 9, 10, 10, 9, 9, 10, 10, 12, 9, 10, 10, 10, 10, 9, 9, 9, 10, 9, 9, 10, 9, 10, 9, 9, 10, 10, 9, 10, 9, 9, 9, 9, 10, 10, 10, 9, 10, 10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 10, 9, 10, 9, 10, 9, 10, 9, 9, 10, 10, 9, 9, 10, 10, 9, 10, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 10, 9, 9, 10, 9, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 10, 10, 10, 10, 9, 10, 9, 10, 10, 10, 9, 9, 9, 10, 9, 9, 8, 9, 9, 9, 9, 9, 10, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 9, 9, 10, 9, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 9, 10, 10, 9, 9, 9, 10, 10, 10, 9, 10, 9, 9, 10, 10, 9, 10, 9, 9, 9, 9, 10, 9, 9, 8, 9, 8, 9, 10, 10, 10, 9, 9, 10, 10, 9, 9, 9, 9, 10, 9, 9, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 10, 8, 10, 10, 9, 10, 9, 10, 10, 9, 10, 9, 9, 9, 9, 10, 9, 9, 10, 9, 10, 9, 9, 10, 10, 10, 10, 10, 9, 10, 9, 9, 9, 10, 9, 10, 10, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 10, 10, 9, 10, 10, 9, 10, 9, 9, 8, 9, 10, 9, 9, 10, 9, 9, 9, 10, 10, 9, 10, 9, 10, 9, 10, 9, 9, 9, 10, 9, 9, 10, 10, 9, 9, 10, 10, 9, 9, 9, 10, 9, 10, 10, 9, 10, 10, 10, 9, 10, 9, 9, 9, 9, 10, 9, 9, 9, 10, 9, 9, 10, 10, 10, 9, 10, 10, 10, 9, 10, 9, 9, 9, 10, 9, 10, 10, 10, 10, 9, 10, 10, 10, 9, 9, 10, 10, 9, 9, 9, 9, 10, 9, 9, 10, 9, 9, 10, 10, 10, 9, 10, 9, 10, 10, 9, 10, 10, 9, 10, 10, 9, 9, 10, 9, 10, 10, 9, 9, 9, 9, 10, 10, 9, 10, 9, 10, 10, 9, 10, 9, 9, 10, 10, 9, 9, 9, 10, 9, 10, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 10, 9, 9, 9, 9, 9, 10, 9, 10, 10, 10, 10, 10, 9, 10, 10, 10, 9, 10, 10, 9, 9, 9, 10, 10, 9, 9, 9, 10, 9, 9, 9, 8, 9, 9, 10, 10, 9, 10, 11, 11, 9, 9, 10, 9, 10, 10, 9, 9, 10, 10, 9, 10, 10, 9, 9, 9, 9, 10, 10, 10, 10, 9, 9, 9, 9, 9, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9]
    SAM_radii = CV_radii

    projections, image_height, image_width = import_tiff_projections(projections_file_path, NUMBER_OF_PROJECTIONS)
    print(projections[0].shape)
    # plot_raw_sinogram(projections, NUMBER_OF_PROJECTIONS, shift_up, background_value)

    CV_CoM, SAM_CoM = deduce_z_axis_CoM(CV_xy_CoM, CV_radii, SAM_xy_CoM, SAM_radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE)
    
    # Get the rotation axis of the trajectory
    CV_rotation_axis, SAM_rotation_axis = get_rotation_axis([CV_CoM, SAM_CoM])

    # Get the point of rotation of the trajectory
    CV_rotation_point, SAM_rotation_point =  get_rotation_point([CV_CoM, SAM_CoM])

    # Plot the deduced trajectory of the sphere with their axis of rotation
    plot_trajectory(CV_CoM, SAM_CoM, CV_rotation_axis, SAM_rotation_axis, CV_rotation_point, SAM_rotation_point)

    CV_x_poly, SAM_x_poly = x_curve_fitting(CV_CoM, SAM_CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=True)
    CV_y_poly, SAM_y_poly = y_curve_fitting(CV_CoM, SAM_CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=True)

    shift_up = determine_roll(SAM_y_poly, NUMBER_OF_PROJECTIONS, image_height, image_width)

    CV_x_sinogram_array, SAM_x_sinogram_array = x_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS)
    CV_y_sinogram_array, SAM_y_sinogram_array = y_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS)

    background_value = determine_background_value(CV_x_sinogram_array)

    shift_up = 0

    # ADD SINUSODAL FIT


    plot_raw_sinogram(projections, NUMBER_OF_PROJECTIONS, shift_up, background_value)

    plot_sinograms(CV_x_sinogram_array, SAM_x_sinogram_array)
    plot_sinograms(CV_y_sinogram_array, SAM_y_sinogram_array)

    shift = 0

    iradon_reconstruction(CV_y_sinogram_array, SAM_y_sinogram_array, background_value, shift)

    CV_CoM, SAM_CoM = add_missing_projections(CV_x_poly, SAM_x_poly, CV_y_poly, SAM_y_poly, CV_CoM, SAM_CoM, projection_idx, NUMBER_OF_PROJECTIONS)

    CV_x_sinogram_array, SAM_x_sinogram_array = x_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)
    CV_y_sinogram_array, SAM_y_sinogram_array = y_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)

    plot_sinograms(CV_x_sinogram_array, SAM_x_sinogram_array)
    plot_sinograms(CV_y_sinogram_array, SAM_y_sinogram_array)

    iradon_reconstruction(CV_y_sinogram_array, SAM_y_sinogram_array, background_value, shift)

    corrected_CV_projections = correct_data(CV_y_poly, CV_CoM, projections, NUMBER_OF_PROJECTIONS, background_value)
    corrected_SAM_projections = correct_data(SAM_y_poly, SAM_CoM, projections, NUMBER_OF_PROJECTIONS, background_value)

    CV_x_sinogram_array, SAM_x_sinogram_array = x_sinograms([CV_CoM, SAM_CoM], [corrected_CV_projections, corrected_SAM_projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)
    CV_y_sinogram_array, SAM_y_sinogram_array = y_sinograms([CV_CoM, SAM_CoM], [corrected_CV_projections, corrected_SAM_projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)

    plot_sinograms(CV_x_sinogram_array, SAM_x_sinogram_array)
    plot_sinograms(CV_y_sinogram_array, SAM_y_sinogram_array)

    plot_raw_sinogram(corrected_CV_projections, NUMBER_OF_PROJECTIONS, shift_up, background_value)

    iradon_reconstruction(CV_y_sinogram_array, SAM_y_sinogram_array, background_value, shift)

    plot_raw_sinogram(corrected_SAM_projections, NUMBER_OF_PROJECTIONS, shift_up, background_value)

if __name__ == '__main__':
    main()