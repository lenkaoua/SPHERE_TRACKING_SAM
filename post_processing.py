import numpy as np
import matplotlib.pyplot as plt
import functools
import scipy.optimize
import tifffile
import ast
from skimage.transform import iradon

def correct_data(CV_x_poly, CV_y_poly, CV_CoM, projections, NUMBER_OF_PROJECTIONS):

    CV_x_components = [point[0] for point in CV_CoM]

    # Calculate the residuals for CV and SAM curves
    CV_x_residuals = CV_x_components - CV_x_poly(list(range(NUMBER_OF_PROJECTIONS)))

    # Find the index of the point with the maximum residual for both CV and SAM
    max_CV_x_residual_idx = int(max(CV_x_residuals))
    min_CV_x_residual_idx = int(min(CV_x_residuals))

    padded_projections = []  # Initialize an empty list

    for i, projection in enumerate(projections):
        
        extra_top_rows = np.full((max_CV_x_residual_idx, projection.shape[1]), np.nan)
        extra_bottom_rows = np.full((-min_CV_x_residual_idx, projection.shape[1]), np.nan)
        padded_projection = np.vstack([extra_bottom_rows, projection, extra_top_rows])

        padded_projections.append(padded_projection)  # Append the padded projection to the list
    
    # Convert the list of arrays to a single numpy array
    padded_projections = np.array(padded_projections)

    # corrected_projections = []

    corrected_projections = []

    for i, projection in enumerate(padded_projections):

        delta_CV_x = CV_x_components[i] - CV_x_poly(i)
        delta_CV_x = int(delta_CV_x)

        if delta_CV_x > 0:
            
            trimmed_projection = projection[delta_CV_x:]
            extra_top_rows = np.full((delta_CV_x, trimmed_projection.shape[1]), np.nan)
            corrected_projection = np.vstack([trimmed_projection, extra_top_rows])

            corrected_projections.append(corrected_projection)
        
        elif delta_CV_x < 0:
            
            trimmed_projection = projection[:delta_CV_x]
            extra_bottom_rows = np.full((-delta_CV_x, trimmed_projection.shape[1]), np.nan)
            corrected_projection = np.vstack([extra_bottom_rows, trimmed_projection])

            corrected_projections.append(corrected_projection)

        else:

            corrected_projections.append(projection)

    corrected_projections = np.array(corrected_projections)

    return corrected_projections

def iradon_reconstruction(CV_sinogram_array, SAM_sinogram_array):

    # number_of_projections = CV_sinogram_array.size[1]

    theta = np.linspace(0., 180., max(CV_sinogram_array.shape), endpoint=False)

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

def y_curve_fitting(CV_CoM, SAM_CoM, projection_idx, plot=False):

    CV_y_CoM = [y[1] for y in CV_CoM]
    SAM_y_CoM = [y[1] for y in SAM_CoM]

    # Fit a 3rd order polynomial for CV_y_CoM
    CV_coeff = np.polyfit(projection_idx, CV_y_CoM, 3)
    CV_poly = np.poly1d(CV_coeff)

    # Fit a 3rd order polynomial for SAM_y_CoM
    SAM_coeff = np.polyfit(projection_idx, SAM_y_CoM, 3)
    SAM_poly = np.poly1d(SAM_coeff)

    # Generate points for the fitted polynomials
    x_fit = np.linspace(0, max(projection_idx), 100)
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

def x_curve_fitting(CV_CoM, SAM_CoM, projection_idx, plot=False):
    
    CV_x_CoM = [x[0] for x in CV_CoM]
    SAM_x_CoM = [x[0] for x in SAM_CoM]

    # Fit a 1st degree polynomial for CV_y_CoM (linear regression)
    CV_coeff = np.polyfit(projection_idx, CV_x_CoM, 1)
    CV_poly = np.poly1d(CV_coeff)

    # Fit a 1st degree polynomial for SAM_y_CoM (linear regression)
    SAM_coeff = np.polyfit(projection_idx, SAM_x_CoM, 1)
    SAM_poly = np.poly1d(SAM_coeff)

    # Generate points for the fitted polynomials
    x_fit = np.linspace(0, max(projection_idx), 100)
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
    
    # # Reset the axes limits to their default values
    ax.set_xlim(65, 85)
    # ax.set_ylim(-1000, 2000)
    ax.set_zlim(-0000, -300000)

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

    return images

def plot_raw_sinogram(projections, NUMBER_OF_PROJECTIONS):

    sinogram = []

    for i in range(NUMBER_OF_PROJECTIONS):

        projection = projections[i]

        x = projection[:,0]
        sinogram.append(x)

    sinogram = np.transpose(sinogram)

    sinogram = np.roll(sinogram, 64, axis=0)
    

    # Create subplots
    fig, (ax1, ax2)  = plt.subplots(1, 2)

    theta = np.linspace(0., 180., 652, endpoint=False)

    CV_reconstruction_fbp = iradon(-np.log(sinogram/47000), theta=theta, filter_name='ramp')

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

def main():

    PIXEL_SIZE = 1.1e-6 # 1.1 μm
    SOURCE_SAMPLE_DISTANCE = 220e-2 # 220 cm
    SAMPLE_DETECTOR_DISTANCE = 1e-2 # 1 cm
    SPHERE_RADIUS = 25e-6 # 40 μm
    SOURCE_DETECTOR_DISTANCE = SOURCE_SAMPLE_DISTANCE + SAMPLE_DETECTOR_DISTANCE # cm
    NUMBER_OF_PROJECTIONS = 652

    projections_file_path = 'ProjectionsData.tiff'
    data_folder = 'Images'

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

    projections = import_tiff_projections(projections_file_path, NUMBER_OF_PROJECTIONS)
    
    plot_raw_sinogram(projections, NUMBER_OF_PROJECTIONS)

    CV_CoM, SAM_CoM = deduce_z_axis_CoM(CV_xy_CoM, CV_radii, SAM_xy_CoM, SAM_radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE)
    
    # Get the rotation axis of the trajectory
    CV_rotation_axis, SAM_rotation_axis = get_rotation_axis([CV_CoM, SAM_CoM])

    # Get the point of rotation of the trajectory
    CV_rotation_point, SAM_rotation_point =  get_rotation_point([CV_CoM, SAM_CoM])

    # Plot the deduced trajectory of the sphere with their axis of rotation
    plot_trajectory(CV_CoM, SAM_CoM, CV_rotation_axis, SAM_rotation_axis, CV_rotation_point, SAM_rotation_point)
    
    CV_x_poly, SAM_x_poly = x_curve_fitting(CV_CoM, SAM_CoM, projection_idx, plot=True)
    CV_y_poly, SAM_y_poly = y_curve_fitting(CV_CoM, SAM_CoM, projection_idx, plot=True)

    CV_x_sinogram_array, SAM_x_sinogram_array = x_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS)
    CV_y_sinogram_array, SAM_y_sinogram_array = y_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS)

    plot_sinograms(CV_x_sinogram_array, SAM_x_sinogram_array)
    plot_sinograms(CV_y_sinogram_array, SAM_y_sinogram_array)

    iradon_reconstruction(CV_x_sinogram_array, SAM_x_sinogram_array)

    CV_CoM, SAM_CoM = add_missing_projections(CV_x_poly, SAM_x_poly, CV_y_poly, SAM_y_poly, CV_CoM, SAM_CoM, projection_idx, NUMBER_OF_PROJECTIONS)

    CV_x_sinogram_array, SAM_x_sinogram_array = x_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)
    CV_y_sinogram_array, SAM_y_sinogram_array = y_sinograms([CV_CoM, SAM_CoM], [projections, projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)

    plot_sinograms(CV_x_sinogram_array, SAM_x_sinogram_array)
    plot_sinograms(CV_y_sinogram_array, SAM_y_sinogram_array)

    iradon_reconstruction(CV_x_sinogram_array, SAM_x_sinogram_array)

    corrected_CV_projections = correct_data(CV_x_poly, CV_y_poly, CV_CoM, projections, NUMBER_OF_PROJECTIONS)
    corrected_SAM_projections = correct_data(SAM_x_poly, SAM_y_poly, SAM_CoM, projections, NUMBER_OF_PROJECTIONS)

    CV_x_sinogram_array, SAM_x_sinogram_array = x_sinograms([CV_CoM, SAM_CoM], [corrected_CV_projections, corrected_SAM_projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)
    CV_y_sinogram_array, SAM_y_sinogram_array = y_sinograms([CV_CoM, SAM_CoM], [corrected_CV_projections, corrected_SAM_projections], projection_idx, NUMBER_OF_PROJECTIONS, complete=True)


    plot_sinograms(CV_x_sinogram_array, SAM_x_sinogram_array)
    plot_sinograms(CV_y_sinogram_array, SAM_y_sinogram_array)

    iradon_reconstruction(CV_y_sinogram_array, SAM_y_sinogram_array)

if __name__ == '__main__':
    main()