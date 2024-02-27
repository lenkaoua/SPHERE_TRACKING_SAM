import numpy as np
import matplotlib.pyplot as plt
import functools
import scipy.optimize

def plot_x_sinogram(CV_CoM, SAM_CoM, projection_idx):
    
    CV_x_CoM = [x[0] for x in CV_CoM]
    SAM_x_CoM = [x[0] for x in SAM_CoM]

    # Fit a 1st degree polynomial for CV_y_CoM (linear regression)
    coeff_cv = np.polyfit(projection_idx, CV_x_CoM, 1)
    poly_cv = np.poly1d(coeff_cv)

    # Fit a 1st degree polynomial for SAM_y_CoM (linear regression)
    coeff_sam = np.polyfit(projection_idx, SAM_x_CoM, 1)
    poly_sam = np.poly1d(coeff_sam)

    # Generate points for the fitted polynomials
    x_fit = np.linspace(0, max(projection_idx), 100)
    y_fit_cv = poly_cv(x_fit)
    y_fit_sam = poly_sam(x_fit)

    # Create figure and axes objects with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

    # Plot for the first subplot (CV)
    axs[0].scatter(projection_idx, CV_x_CoM, marker='o', color='blue', label='Data')
    axs[0].plot(x_fit, y_fit_cv, 'r-', label='Fitted line (CV)')
    axs[0].set_xlabel('Projection Number')
    axs[0].set_ylabel('CV X-axis')
    axs[0].set_title('CV Sinogram along X-axis')
    axs[0].legend()
    axs[0].grid(True)

    # Plot for the second subplot (SAM)
    axs[1].scatter(projection_idx, SAM_x_CoM, marker='o', color='green', label='Data')
    axs[1].plot(x_fit, y_fit_sam, 'r-', label='Fitted line (SAM)')
    axs[1].set_xlabel('Projection Number')
    axs[1].set_ylabel('SAM X-axis')
    axs[1].set_title('SAM Sinogram along X-axis')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()

def plot_y_sinogram(CV_CoM, SAM_CoM, projection_idx):

    CV_y_CoM = [y[1] for y in CV_CoM]
    SAM_y_CoM = [y[1] for y in SAM_CoM]

    # Fit a 3rd order polynomial for CV_y_CoM
    coeff_cv = np.polyfit(projection_idx, CV_y_CoM, 3)
    poly_cv = np.poly1d(coeff_cv)

    # Fit a 3rd order polynomial for SAM_y_CoM
    coeff_sam = np.polyfit(projection_idx, SAM_y_CoM, 3)
    poly_sam = np.poly1d(coeff_sam)

    # Generate points for the fitted polynomials
    x_fit = np.linspace(0, max(projection_idx), 100)
    y_fit_cv = poly_cv(x_fit)
    y_fit_sam = poly_sam(x_fit)

    # Create figure and axes objects with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

    # Plot for the first subplot (CV)
    axs[0].scatter(projection_idx, CV_y_CoM, marker='o', color='blue', label='Data')
    axs[0].plot(x_fit, y_fit_cv, 'r-', label='Fitted curve (CV)')
    axs[0].set_xlabel('Projection Number')
    axs[0].set_ylabel('CV Y-axis')
    axs[0].set_title('CV Sinogram along Y-axis')
    axs[0].legend()
    axs[0].grid(True)

    # Plot for the second subplot (SAM)
    axs[1].scatter(projection_idx, SAM_y_CoM, marker='o', color='green', label='Data')
    axs[1].plot(x_fit, y_fit_sam, 'r-', label='Fitted curve (SAM)')
    axs[1].set_xlabel('Projection Number')
    axs[1].set_ylabel('SAM Y-axis')
    axs[1].set_title('SAM Sinogram along Y-axis')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

def plot_trajectory(CV_CoM, SAM_CoM, CV_rotation_axis, SAM_rotation_axis, CV_rotation_point, SAM_rotation_point):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title('Trajectory of the Sphere')

    CV_x_CoM = [x[0] for x in CV_CoM]
    CV_y_CoM = [y[1] for y in CV_CoM]
    CV_z_CoM = [z[2] for z in CV_CoM]

    SAM_x_CoM = [x[0] for x in SAM_CoM]
    SAM_y_CoM = [y[1] for y in SAM_CoM]
    SAM_z_CoM = [z[2] for z in SAM_CoM]

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

    print(f'\nOpen CV CoMs: {CV_CoM}\nOpen CV Radii: {CV_radii}\nSAM CoMs: {SAM_CoM}\nSAM Radii: {SAM_radii}')

    return CV_CoM, SAM_CoM

def main():

    PIXEL_SIZE = 1.1e-6 # 1.1 μm
    SOURCE_SAMPLE_DISTANCE = 220e-2 # 220 cm
    SAMPLE_DETECTOR_DISTANCE = 1e-2 # 1 cm
    SPHERE_RADIUS = 25e-6 # 40 μm
    SOURCE_DETECTOR_DISTANCE = SOURCE_SAMPLE_DISTANCE + SAMPLE_DETECTOR_DISTANCE # cm

    CV_xy_CoM = [[77, 233], [77, 275], [74, 322], [73, 381], [73, 448], [75, 522], [72, 681], [74, 767], [76, 850], [72, 923], [74, 1002], [74, 1069], [73, 1132], [77, 1180], [75, 1218], [75, 1246], [72, 1260], [77, 1265], [77, 1232]]
    SAM_xy_CoM = [[75.5, 232.5], [75.0, 275.0], [74.5, 323.5], [74.0, 382.0], [75.0, 446.0], [74.5, 520.5], [74.5, 683.5], [75.0, 766.0], [74.5, 848.5], [75.0, 926.0], [75.0, 1001.0], [75.0, 1069.0], [75.0, 1130.0], [75.0, 1179.0], [75.0, 1219.0], [75.5, 1247.5], [74.5, 1262.5], [75.0, 1266.0], [74.5, 1233.5]]
    
    projection_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    CV_radii = [21, 22, 21, 21, 21, 22, 22, 22, 21, 22, 21, 21, 21, 22, 22, 21, 21, 19, 21, 22]
    SAM_radii = [20.5, 22.0, 21.5, 21.0, 21.0, 21.5, 21.5, 21.0, 20.5, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.5, 20.5, 20.0, 21.0, 20.5]

    NUMBER_OF_PROJECTIONS = max(projection_idx)

    CV_CoM, SAM_CoM = deduce_z_axis_CoM(CV_xy_CoM, CV_radii, SAM_xy_CoM, SAM_radii, SPHERE_RADIUS, SOURCE_DETECTOR_DISTANCE, PIXEL_SIZE)

    # Get the rotation axis of the trajectory
    CV_rotation_axis, SAM_rotation_axis = get_rotation_axis([CV_CoM, SAM_CoM])

    # Get the point of rotation of the trajectory
    CV_rotation_point, SAM_rotation_point =  get_rotation_point([CV_CoM, SAM_CoM])

    # Plot the deduced trajectory of the sphere with their centre and axis of rotation
    plot_trajectory(CV_CoM, SAM_CoM, CV_rotation_axis, SAM_rotation_axis, CV_rotation_point, SAM_rotation_point)

    plot_x_sinogram(CV_CoM, SAM_CoM, projection_idx)

    plot_y_sinogram(CV_CoM, SAM_CoM, projection_idx)

if __name__ == '__main__':
    main()