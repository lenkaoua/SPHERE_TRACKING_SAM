import os
import sys
from PIL import Image
import random

def shift_image_columns(image):
    width, height = image.size
    pixels = image.load()

    for x in range(width):
        # Generate a random shift amount for each column
        shift_amount = random.randint(-40, 40)  # Adjust the range as needed
        # Shift the column by the random amount
        shifted_column = [pixels[x, (y + shift_amount) % height] for y in range(height)]
        for y in range(height):
            pixels[x, y] = shifted_column[y]

# Load the image
image = Image.open("Brain Sinogram.png")

# Perform the column shifting
shift_image_columns(image)

# Save the modified image
image.save("shifted_image.png")
