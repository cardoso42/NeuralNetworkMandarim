from PIL import Image
import os
import numpy as np
import sys

def image_to_binary(image_path, threshold=128):
    """
    Convert an image to binary format using a threshold.
    
    Parameters:
    - image_path: Path to the input image file.
    - threshold: Threshold value for converting pixel values to binary. Default is 128.
    
    Returns:
    - Binary NumPy array representing the image.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    binary_array = np.array(image.point(lambda x: 0 if x < threshold else 255), dtype=np.uint8)
    binary_array[binary_array < 128] = 0
    binary_array[binary_array >= 128] = 1
    return binary_array

def convert_images_to_binary(input_dir, output_dir, threshold=128):
    """
    Convert all images in a directory to binary format and save them to another directory.
    
    Parameters:
    - input_dir: Input directory containing images.
    - output_dir: Output directory to save binary images.
    - threshold: Threshold value for converting pixel values to binary. Default is 128.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        binary_array = image_to_binary(input_path, threshold)
        binary_image = Image.fromarray(binary_array * 255)  # Convert binary array back to image
        binary_image.save(output_path)

if __name__ == "__main__":
        # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)

    # Get input and output folders from command-line arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    threshold_value = 128  # Adjust this threshold as needed

    convert_images_to_binary(input_folder, output_folder, threshold_value)
