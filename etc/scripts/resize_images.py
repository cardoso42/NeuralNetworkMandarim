from PIL import Image
import os
import sys

def resize_images(input_folder, output_folder, target_size=(28, 28)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the image using Pillow
        with Image.open(input_path) as img:
            # Resize the image
            resized_img = img.resize(target_size)

            # Save the resized image to the output folder
            resized_img.save(output_path)

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)

    # Get input and output folders from command-line arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Call the resize_images function
    resize_images(input_folder, output_folder)

