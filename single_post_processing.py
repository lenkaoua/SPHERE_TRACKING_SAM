import numpy as np
import matplotlib.pyplot as plt
import functools
import scipy.optimize
import tifffile
import ast
from skimage.transform import iradon

def plot_results(row_sinogram, CoM_sinogram, iradon_reconstruction):

    # Create subplots
    fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, figsize=(12, 4))

    # Plot CoM sinogram
    cax1 = ax1.imshow(CoM_sinogram, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Projection Number')
    ax1.set_ylabel('Y-Values')
    ax1.set_title('Sphere Tracking Sinogram')
    fig.colorbar(cax1, ax=ax1, shrink=0.8)

    # Plot fixed row sinogram
    cax2 = ax2.imshow(row_sinogram, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Projection Number')
    ax2.set_ylabel('Y-Values')
    ax2.set_title('Fixed Row Sinogram')
    fig.colorbar(cax2, ax=ax2, shrink=0.8)

    cax3 = ax3.imshow(iradon_reconstruction, cmap=plt.cm.Greys_r)
    ax3.set_title('Filtered Back Projection Reconstruction')
    fig.colorbar(cax3, ax=ax3, shrink=0.8)

    plt.tight_layout()
    plt.show()

def correct_data(y_poly, CoM, projections, reverse=False):

    corrected_projections = []

    count = 0

    for i, projection in enumerate(projections):

        y_CoM = CoM[i][1]
        expected_y_CoM = y_poly(count)
        count += 1

        # Calculate vertical shift
        y_shift = int(expected_y_CoM - y_CoM)

        # Creating a new array filled with ones, representing the background
        corrected_projection = np.full_like(projection, 0, dtype=np.float64)

        if not reverse:
            if y_shift >= 1:  # Invert: Shift the projection up
                corrected_projection[int(y_shift):, :] = projection[:-int(y_shift) or None, :]
            elif y_shift <= 1:  # Invert: Shift the projection down
                corrected_projection[:int(y_shift) or None, :] = projection[-int(y_shift):, :]
        else:
            if y_shift >= 1:  # Invert: Shift the projection up
                corrected_projection[:int(-y_shift) or None, :] = projection[int(y_shift):, :]
            elif y_shift <= 1:  # Invert: Shift the projection down
                corrected_projection[int(-y_shift):, :] = projection[:int(y_shift) or None, :]

        corrected_projections.append(corrected_projection)
    
    return corrected_projections

def iradon_reconstruction(row_sinogram, DEGREES):

    number_of_projections = row_sinogram.shape[1]

    theta = np.linspace(0, DEGREES, int(number_of_projections), endpoint=False)

    iradon_reconstruction = iradon(row_sinogram, theta=theta, filter_name='ramp')

    return iradon_reconstruction

def deduce_missing_CoM(x_poly, y_poly, CoM, projection_idx, NUMBER_OF_PROJECTIONS):

    for i in range(NUMBER_OF_PROJECTIONS):

        if i not in projection_idx:
            x_CoM = x_poly(i)
            y_CoM = y_poly(i)
            z_CoM = np.nan

            CoM.insert(i, [x_CoM, y_CoM, z_CoM])

    return CoM

def y_sinograms(CoM, projections, projection_idx, complete=False):

    y_sinogram = []

    if complete == True:
        for i, projection in enumerate(projections):

            y_CoM = int(CoM[i][0])
            slice = projection[:,y_CoM]
            y_sinogram.append(slice)

    elif not complete:
        for i, idx in enumerate(projection_idx):
    
            projection = projections[idx]
            y_CoM = int(CoM[i][0])
            slice = projection[:,y_CoM]
            y_sinogram.append(slice)

    elif complete == 'Fixed Row Sinogram':
        for projection in projections:

            slice = projection[:,20]
            y_sinogram.append(slice)

    y_sinogram = np.transpose(y_sinogram)

    return y_sinogram

def x_sinograms(CoM, projections, projection_idx, complete=False):

    x_sinogram = []
    
    if complete:
        for i, projection in enumerate(projections):

            x_CoM = int(CoM[i][1])
            slice = projection[x_CoM,:]
            x_sinogram.append(slice)

    else:
        for i, idx in enumerate(projection_idx):

            projection = projections[idx]
            x_CoM = int(CoM[i][1])
            slice = projection[x_CoM,:]
            x_sinogram.append(slice)

    x_sinogram = np.transpose(x_sinogram)

    return x_sinogram

def plot_sinograms(CV_sinogram, SAM_sinogram):

    # Create subplots
    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(8, 4))

    # Plot CV_sinogram
    cax1 = ax1.imshow(CV_sinogram, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Projection Number')
    ax1.set_ylabel('Y-Values')
    ax1.set_title('Sphere Tracking Y-Sinogram')
    fig.colorbar(cax1, ax=ax1, shrink=0.8)

    # Plot SAM_sinogram
    cax2 = ax2.imshow(SAM_sinogram, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Projection Number')
    ax2.set_ylabel('X-Values')
    ax2.set_title('Sphere Tracking X-Sinogram')
    fig.colorbar(cax2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.show()

def y_curve_fitting(CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=True):

    y_CoM = [y[1] for y in CoM]

    # Fit a 3rd order polynomial for CV_y_CoM
    coeff = np.polyfit(projection_idx, y_CoM, 12)
    y_poly = np.poly1d(coeff)

    # Generate points for the fitted polynomials
    points = np.linspace(0, NUMBER_OF_PROJECTIONS, 100)
    y_fit = y_poly(points)

    if plot:

        # Create figure and axes objects with two subplots
        fig, ax = plt.subplots(1, 1)  # 1 row, 2 columns

        # Plot for the first subplot (CV)
        ax.scatter(projection_idx, y_CoM, marker='o', color='orange', label='Data', s=3)
        ax.plot(points, y_fit, 'k-', label='Fitted curve (CV)', linewidth=1)
        ax.set_xlabel('Projection Number')
        ax.set_ylabel('CV Y-axis')
        ax.set_title('CV Sinogram along Y-axis')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return y_poly

def x_curve_fitting(CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=True):
    
    x_CoM = [x[0] for x in CoM]

    # Fit a 1st degree polynomial
    coeff = np.polyfit(projection_idx, x_CoM, 1)
    x_poly = np.poly1d(coeff)

    # Generate points for the fitted polynomials
    points = np.linspace(0, NUMBER_OF_PROJECTIONS, 100)
    x_fit = x_poly(points)

    if plot:
        
        # Create figure and axes objects with two subplots
        fig, ax = plt.subplots(1, 1)  # 1 row, 2 columns

        # Plot for the first subplot (CV)
        ax.scatter(projection_idx, x_CoM, marker='o', color='orange', label='Data', s=6)
        ax.plot(points, x_fit, 'k-', label='Fitted line (CV)')
        ax.set_xlabel('Projection Number')
        ax.set_ylabel('CV X-axis')
        ax.set_title('CV Sinogram along X-axis')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    return x_poly

def plot_trajectory(CoM, rotation_axis, rotation_point):

    x_CoM = [x[0] for x in CoM]
    y_CoM = [y[1] for y in CoM]
    z_CoM = [z[2] for z in CoM]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the deduced positions of the sphere in red
    ax.scatter(x_CoM, y_CoM, z_CoM, c='purple', marker='o', label='CV CoM')

    # Plot the deduced point of rotation
    ax.scatter(rotation_point[0], rotation_point[1], rotation_point[2], c='black', marker='x', label='CV Point of Rotation')

    # Plot the deduced rotation axis
    ax.quiver(rotation_point[0], rotation_point[1], rotation_point[2], rotation_axis[0], rotation_axis[2], rotation_axis[1], length=50, color='black', label='CV Axis of Rotation')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory of the Sphere')

    # Rotate the view
    ax.view_init(elev=25, azim=45, vertical_axis='y')

    ax.legend()
    plt.show()
    plt.close(fig)

def get_rotation_point(CoM):

    # Convert points to a NumPy array
    points_array = np.array(CoM)

    # Calculate the average of each dimension
    rotation_point = np.mean(points_array, axis=0)

    print(f'\nPoint of rotation: {(rotation_point[0], rotation_point[1], rotation_point[2])}')

    return rotation_point

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

def get_rotation_axis(CoM):

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

    print(f'\nAxis of rotation: {(rotation_axis[0], rotation_axis[1], rotation_axis[2])}')

    return rotation_axis

def deduce_z_axis_CoM(xy_CoM, radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE):

    CoM = []

    # Convert the source to detector distance and sphere radius to pixel dimensions
    SOURCE_DETECTOR_DISTANCE = SOURCE_DETECTOR_DISTANCE / PIXEL_SIZE
    SPHERE_RADIUS = SPHERE_RADIUS / PIXEL_SIZE
    
    for i in range(len(xy_CoM)):
        
        magnification = SPHERE_RADIUS / radii[i]
        z_CoM = (SOURCE_DETECTOR_DISTANCE / magnification) - SOURCE_DETECTOR_DISTANCE

        CoM.append([xy_CoM[i][0], xy_CoM[i][1], z_CoM])

    return CoM

def log_correction(projections):

    projections = -np.log(projections)

    return projections

def flat_field_correction(projections, background_value):

    projections = projections / background_value

    return projections

def get_background_value(projections):

    sinogram = []

    for projection in projections:

        col = projection[:,20]
        sinogram.append(col)

    sinogram = np.transpose(sinogram)

    background_value = np.average(sinogram[1,:])

    return background_value

def shift_projections(projections, shift=0):
    
    projections = np.roll(projections, shift, axis=1)

    return projections

def import_tiff_projections(file_path, NUMBER_OF_PROJECTIONS):

    all_projections = tifffile.imread(file_path)

    # Calculate the total number of images
    num_projections = len(all_projections)

    # Calculate the spacing between projections to select approximately 100 equally spaced images
    indices = np.linspace(0, num_projections - 1, NUMBER_OF_PROJECTIONS, dtype=int)
    
    images = all_projections[indices]

    return images

def import_text_outputs(data_folder, invert=False):

    with open(f'{data_folder}/xy_CoM.txt', 'r') as file:
        data = file.read()
        xy_CoM = ast.literal_eval(data)
    
    with open(f'{data_folder}/radii.txt', 'r') as file:
        data = file.read()
        radii = ast.literal_eval(data)

    with open(f'{data_folder}/projection_idx.txt', 'r') as file:
        data = file.read()
        projection_idx = ast.literal_eval(data)

    if invert:
        xy_CoM = [[sublist[1], sublist[0]] for sublist in xy_CoM]

    return xy_CoM, radii, projection_idx

def main():

    PIXEL_SIZE = 50e-6 # 1.1 μm
    SOURCE_SAMPLE_DISTANCE = 1 # 220 cm
    SAMPLE_DETECTOR_DISTANCE = 1 # 1 cm
    SPHERE_RADIUS = 25e-6 # 40 μm
    SOURCE_DETECTOR_DISTANCE = SOURCE_SAMPLE_DISTANCE + SAMPLE_DETECTOR_DISTANCE # cm
    NUMBER_OF_PROJECTIONS = 652
    DEGREES = 360

    projections_file_path = 'TiffStack.tif'
    data_folder = 'Data Folder'

    CoM, radii, projection_idx = import_text_outputs(data_folder, invert=False)

    # Get the deduced z-coordinate for each projection
    CoM = deduce_z_axis_CoM(CoM, radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE)
    # Get the rotation axis of the trajectory
    rotation_axis = get_rotation_axis(CoM)
    # Get the point of rotation of the trajectory
    rotation_point =  get_rotation_point(CoM)
    # Plot the deduced trajectory of the sphere with their axis of rotation
    plot_trajectory(CoM, rotation_axis, rotation_point)

    # Import all projections
    projections = import_tiff_projections(projections_file_path, NUMBER_OF_PROJECTIONS)

    # Shift the projections horizontally to centre them
    projections = shift_projections(projections, shift=0)
    # Get the background value
    background_value = get_background_value(projections)
    # Apply flat field correction using the obtained background value
    projections = flat_field_correction(projections, background_value)
    # Apply the logarithm to the projections
    projections = log_correction(projections)

    # Obtain the x and y polynomial fit functions
    x_poly = x_curve_fitting(CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=True)
    y_poly = y_curve_fitting(CoM, projection_idx, NUMBER_OF_PROJECTIONS, plot=True)

    # Obtain the x and y sinograms of the sphere tracking
    x_CoM_raw_sinogram = x_sinograms(CoM, projections, projection_idx)
    y_CoM_raw_sinogram = y_sinograms(CoM, projections, projection_idx)

    # Plot the raw sphere tracking sinograms
    plot_sinograms(y_CoM_raw_sinogram, x_CoM_raw_sinogram)

    # Deduce the missing centre of mass coordinates using the x and y polynomial fits
    CoM = deduce_missing_CoM(x_poly, y_poly, CoM, projection_idx, NUMBER_OF_PROJECTIONS)

    # Obtain the complete x and y sinograms of the sphere tracking
    x_CoM_complete_sinogram = x_sinograms(CoM, projections, projection_idx, complete=True)
    y_CoM_complete_sinogram = y_sinograms(CoM, projections, projection_idx, complete=True)

    # Plot the complete sphere tracking sinograms
    plot_sinograms(y_CoM_complete_sinogram, x_CoM_complete_sinogram)

    # Obtain the sinogram for a fixed row in the projections
    y_raw_row_sinogram = y_sinograms(CoM, projections, projection_idx, complete='Fixed Row Sinogram')
    # Reconstruct the fixed-row sinogram
    raw_reconstruction = iradon_reconstruction(y_raw_row_sinogram, DEGREES)
    # Plot the complete sphere tracking sinogram, and the the raw fixed-row sinogram with its reconstruction
    plot_results(y_raw_row_sinogram, y_CoM_complete_sinogram, raw_reconstruction)

    # Correct the projections using the y polynomial fit
    corrected_projections = correct_data(y_poly, CoM, projections, reverse=False)

    # Obtain the x and y sinograms of the corrected sphere tracking
    x_CoM_corrected_sinogram = x_sinograms(CoM, corrected_projections, projection_idx, complete=True)
    y_CoM_corrected_sinogram = y_sinograms(CoM, corrected_projections, projection_idx, complete=True)
    
    # Plot the corrected sphere tracking sinograms
    plot_sinograms(y_CoM_corrected_sinogram, x_CoM_corrected_sinogram)

    # Obtain the sinogram for a fixed row in the projections
    y_corrected_row_sinogram = y_sinograms(CoM, corrected_projections, projection_idx, complete='Fixed Row Sinogram')
    # Reconstruct the fixed-row sinogram
    corrected_reconstruction = iradon_reconstruction(y_corrected_row_sinogram, DEGREES)
    # Plot the corrected sphere tracking sinogram, and the the corrected fixed-row sinogram with its reconstruction
    plot_results(y_corrected_row_sinogram, y_CoM_corrected_sinogram, corrected_reconstruction)

if __name__ == '__main__':
    main()