import numpy as np
import matplotlib.pyplot as plt
import functools
import scipy.optimize
import tifffile
import ast
from skimage.transform import iradon
from scipy.optimize import curve_fit

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

def correct_data(y_sinusoidal_fit_params, CoM, projections, invert=False, plot=False):

    A, B, C, D = y_sinusoidal_fit_params

    corrected_projections = []
    
    y_shifts = []

    count = 0

    for i, projection in enumerate(projections):
    
        y_CoM = CoM[i][1]
        expected_y_CoM = sinusoidal_func(i, A, B, C, D)
        count += 1

        # Calculate vertical shift
        y_shift = int(expected_y_CoM - y_CoM)

        y_shifts.append(expected_y_CoM - y_CoM)
    
        # Creating a new array filled with ones, representing the background
        corrected_projection = np.full_like(projection, 0, dtype=np.float64)

        if not invert:
            if y_shift >= 1:
                corrected_projection[y_shift:, :] = projection[:-y_shift or None, :]
            elif y_shift <= 1:
                corrected_projection[:y_shift or None, :] = projection[-y_shift:, :]
        else:
            if y_shift >= 1:
                corrected_projection[:-y_shift or None, :] = projection[y_shift:, :]
            elif y_shift <= 1:
                corrected_projection[-y_shift:, :] = projection[:y_shift or None, :]

        corrected_projections.append(corrected_projection)

    num_projections = len(CoM)

    if plot:

        filtered_y_shifts = [y for y in y_shifts[:num_projections] if y > 0.5 or y < -0.5]
        shift_indexes = [i for i, y in enumerate(y_shifts[:num_projections]) if y > 0.5 or y < -0.5]

        fig, ax = plt.subplots()

        # Create a scatter plot
        ax.scatter(shift_indexes, filtered_y_shifts, marker='x', label='Sphere Centre of Mass')
        ax.axhline(y=0, color='black', linestyle='-', label='Correction Error')

        # Add arrows for each non-zero point
        for i, y in zip(shift_indexes, filtered_y_shifts):
            if y > 0:
                ax.arrow(i, y, 0, -y + 0.5, head_width=1, head_length=0.5, fc='blue', ec='blue', overhang=0.7)
            else:
                ax.arrow(i, y, 0, -y - 0.5, head_width=1, head_length=0.5, fc='red', ec='red', overhang=0.7)

        # Add label for arrows
        ax.arrow(0, 0, 0, 0, head_width=0, head_length=0, fc='blue', ec='blue', overhang=0.7, label='Down-Shifting Projections')
        ax.arrow(0, 0, 0, 0, head_width=0, head_length=0, fc='red', ec='red', overhang=0.7, label='Up-Shifting Projections')

        ax.set_title(f'Shifting Corrections for the First {num_projections} Projections')
        ax.legend()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='-')
        ax.xaxis.grid(color='gray', linestyle='-')

        plt.show()

    return corrected_projections

def iradon_reconstruction(row_sinogram, DEGREES):

    number_of_projections = row_sinogram.shape[1]

    theta = np.linspace(0, DEGREES, int(number_of_projections), endpoint=False)

    iradon_reconstruction = iradon(row_sinogram, theta=theta, filter_name='ramp')

    # Extract the middle row index
    middle_row_index = iradon_reconstruction.shape[0] // 2

    # Extract the middle row of the matrix
    middle_row = iradon_reconstruction[middle_row_index, :]

    # Plot the intensity values of the middle row
    plt.plot(middle_row)
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.title('Intensity Plot of Middle Row')
    plt.grid(True)
    plt.show()

    return iradon_reconstruction

def deduce_missing_CoM(y_sinusoidal_fit_params, average_x_CoM, CoM, projection_idx, NUMBER_OF_PROJECTIONS):

    A, B, C, D = y_sinusoidal_fit_params

    for i in range(NUMBER_OF_PROJECTIONS):

        if i not in projection_idx:
            x_CoM = average_x_CoM
            y_CoM = sinusoidal_func(i, A, B, C, D)
            z_CoM = np.nan

            CoM.insert(i, [x_CoM, y_CoM, z_CoM])

    return CoM

def plot_sinogram(sinogram):

    fig, ax  = plt.subplots(1, 1)

    cax1 = ax.imshow(sinogram, aspect='auto', cmap='viridis')
    ax.set_xlabel('Projection Number')
    ax.set_ylabel('Y-Values')
    ax.set_title('Sphere Tracking Sinogram')
    fig.colorbar(cax1, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()

def sinogram(CoM, projections, projection_idx, type='not complete'):

    sinogram = []

    if type == 'complete':
        for i, projection in enumerate(projections):

            y_CoM = int(CoM[i][0])
            slice = projection[:,y_CoM]
            sinogram.append(slice)

    elif type == 'not complete':
        for i, idx in enumerate(projection_idx):
    
            projection = projections[idx]
            y_CoM = int(CoM[i][0])
            slice = projection[:,y_CoM]
            sinogram.append(slice)

    else:
        for projection in projections:

            slice = projection[:,type]
            sinogram.append(slice)

    sinogram = np.transpose(sinogram)

    return sinogram

# Define the fitting function
def sinusoidal_func(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def y_sinusoidal_curve_fitting(CoM, projection_idx, plot=True):

    projection_idx = np.array(projection_idx)
    y_CoM = np.array([y[1] for y in CoM])

    # Initial guesses for parameters
    initial_guesses = [250, 2*np.pi/600, np.pi, 0]

    # Perform the curve fitting
    y_sinusoidal_fit_params, covariance = curve_fit(sinusoidal_func, projection_idx, y_CoM, p0=initial_guesses)

    # Extract the fitted parameters
    A, B, C, D = y_sinusoidal_fit_params

    y_fit = sinusoidal_func(projection_idx, A, B, C, D)

    if plot:
         
        # Create figure and axes objects with two subplots
        fig, ax = plt.subplots(1, 1)  # 1 row, 2 columns

        ax.scatter(projection_idx, y_CoM, marker='o', color='orange', label='Sphere Centre of Mass', s=3)
        ax.plot(projection_idx, y_fit, 'k-', label='Sinusoidal Fitted Curve', linewidth=1)
        ax.set_xlabel('Projection Number')
        ax.set_ylabel('Centre of Mass Y-Coordinate')
        ax.set_title("Y-Coordinates of the Sphere's Centre of Mass and Sinusoidal Curve Fitting vs Projection Number")
        ax.legend()
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='-')
        ax.xaxis.grid(color='gray', linestyle='-')

        plt.tight_layout()
        plt.show()

    return y_sinusoidal_fit_params

def x_curve_fitting(CoM, projection_idx, plot=True):
    
    x_CoM = [x[0] for x in CoM]
    
    average_x_CoM = int(np.average(x_CoM))

    if plot:
        
        fig, ax = plt.subplots(1, 1)

        ax.scatter(projection_idx, x_CoM, marker='o', color='purple', label='Sphere Centre of Mass', s=6)
        ax.set_xlabel('Projection Number')
        ax.set_ylabel('Centre of Mass X-Coordinate')
        ax.set_title("X-Coordinates of the Sphere's Centre of Mass vs Projection Number")
        ax.legend()
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='-')
        ax.xaxis.grid(color='gray', linestyle='-')

        # Adjust layout
        plt.tight_layout()
        plt.show()

    return average_x_CoM

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

def plot_trajectory(CoM, rotation_axis, rotation_point):

    x_CoM = [x[0] for x in CoM]
    y_CoM = [y[1] for y in CoM]
    z_CoM = [z[2] for z in CoM]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the deduced positions of the sphere in red
    ax.scatter(x_CoM, y_CoM, z_CoM, c='purple', marker='o', label='Sphere Centre of Mass')

    # Plot the deduced point of rotation
    ax.scatter(rotation_point[0], rotation_point[1], rotation_point[2], c='black', marker='x', label='Point of Rotation')

    # Plot the deduced rotation axis
    ax.quiver(rotation_point[0], rotation_point[1], rotation_point[2], rotation_axis[0], rotation_axis[2], rotation_axis[1], length=50, color='black', label='Axis of Rotation')

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

    fixed_row_projection = 20

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
    average_x_CoM = x_curve_fitting(CoM, projection_idx, plot=True)
    y_sinusoidal_fit_params = y_sinusoidal_curve_fitting(CoM, projection_idx, plot=True)

    # Obtain the sinogram of the sphere tracking
    CoM_raw_sinogram = sinogram(CoM, projections, projection_idx, type='not complete')

    # Plot the raw sphere tracking sinograms
    plot_sinogram(CoM_raw_sinogram)

    # Deduce the missing centre of mass coordinates using the x and y polynomial fits
    CoM = deduce_missing_CoM(y_sinusoidal_fit_params, average_x_CoM, CoM, projection_idx, NUMBER_OF_PROJECTIONS)

    # Obtain the complete sinograms of the sphere tracking
    CoM_complete_sinogram = sinogram(CoM, projections, projection_idx, type='complete')

    # Plot the complete sphere tracking sinograms
    plot_sinogram(CoM_complete_sinogram)

    # Obtain the sinogram for a fixed row in the projections
    raw_row_sinogram = sinogram(CoM, projections, projection_idx, type=fixed_row_projection)
    # Reconstruct the fixed-row sinogram
    raw_reconstruction = iradon_reconstruction(raw_row_sinogram, DEGREES)
    # Plot the complete sphere tracking sinogram, and the the raw fixed-row sinogram with its reconstruction
    plot_results(raw_row_sinogram, CoM_complete_sinogram, raw_reconstruction)

    # Correct the projections using the y polynomial fit
    corrected_projections = correct_data(y_sinusoidal_fit_params, CoM, projections, invert=False, plot=True)

    # Obtain the sinogram of the corrected sphere tracking
    CoM_corrected_sinogram = sinogram(CoM, corrected_projections, projection_idx, type='complete')
    
    # Plot the corrected sphere tracking sinograms
    plot_sinogram(CoM_corrected_sinogram)

    # Obtain the sinogram for a fixed row in the projections
    corrected_row_sinogram = sinogram(CoM, corrected_projections, projection_idx, type=fixed_row_projection)
    # Reconstruct the fixed-row sinogram
    corrected_reconstruction = iradon_reconstruction(corrected_row_sinogram, DEGREES)
    # Plot the corrected sphere tracking sinogram, and the the corrected fixed-row sinogram with its reconstruction
    plot_results(corrected_row_sinogram, CoM_corrected_sinogram, corrected_reconstruction)

if __name__ == '__main__':
    main()