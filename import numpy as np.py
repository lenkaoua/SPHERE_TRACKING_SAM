import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import ast 


data_folder = 'Images 2'

with open(f'{data_folder}/projection_idx.txt', 'r') as file:
        # Read the contents of the file
        data = file.read()
        # Use ast.literal_eval() to convert the string to a Python object (list of lists)
        projection_idx = ast.literal_eval(data)

with open(f'{data_folder}/SAM_xy_CoM.txt', 'r') as file:
    # Read the contents of the file
    data = file.read()
    # Use ast.literal_eval() to convert the string to a Python object (list of lists)
    SAM_xy_CoM = ast.literal_eval(data)


xy = [[sublist[1], sublist[0]] for sublist in SAM_xy_CoM]

xy = [y[0] for y in SAM_xy_CoM]

SAM_xy_CoM = []
values = []

for i in range(1):
    SAM_xy_CoM.append(xy)  # Appending the entire xy list (presumably a list of numbers)
    values.append(projection_idx)

# Flatten SAM_xy_CoM to remove the brackets
SAM_xy_CoM = [item for sublist in SAM_xy_CoM for item in sublist]
values = [item for sublist in values for item in sublist]

# Fit a 1st degree polynomial for CV_y_CoM (linear regression)
CV_coeff = np.polyfit(values, SAM_xy_CoM, 10)
CV_poly = np.poly1d(CV_coeff)

# Fit a 1st degree polynomial for SAM_y_CoM (linear regression)
SAM_coeff = np.polyfit(values, SAM_xy_CoM, 10)
SAM_poly = np.poly1d(SAM_coeff)

# Generate points for the fitted polynomials
x_fit = np.linspace(0, max(values), 100)
CV_y_fit = CV_poly(x_fit)
SAM_y_fit = SAM_poly(x_fit)

# Create figure and axes objects with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Plot for the first subplot (CV)
ax1.scatter(values, SAM_xy_CoM, marker='o', color='red', label='Data', s=10)
ax1.plot(x_fit, CV_y_fit, 'k-', label='Fitted line (CV)')
ax1.set_xlabel('Projection Number')
ax1.set_ylabel('CV X-axis')
ax1.set_title('CV Sinogram along X-axis')
ax1.legend()
ax1.grid(True)

# Plot for the second subplot (SAM)
ax2.scatter(values, SAM_xy_CoM, marker='o', color='blue', label='Data', s=10)
ax2.plot(x_fit, SAM_y_fit, 'k-', label='Fitted line (SAM)')
ax2.set_xlabel('Projection Number')
ax2.set_ylabel('SAM X-axis')
ax2.set_title('SAM Sinogram along X-axis')
ax2.legend()
ax2.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()