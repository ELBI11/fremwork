import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from urllib.request import urlretrieve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducible results
np.random.seed(42)

def image_load(image_path, channel):
    """
    Load an image with specified channel configuration
    Args:
        image_path (str): Path to the image file
        channel (int): 1 for grayscale, 3 for RGB
    Returns:
        numpy.ndarray: Loaded image
    """
    assert os.path.exists(image_path), f"Error: Image file {image_path} not found"
    
    if channel == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Error: Unable to load grayscale image"
        assert len(image.shape) == 2, "Error: Expected grayscale image (2D array)"
    elif channel == 3:
        image = cv2.imread(image_path)
        assert image is not None, "Error: Unable to load RGB image"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3 and image.shape[2] == 3, "Error: Expected RGB image (3D array with 3 channels)"
    else:
        raise ValueError("Channel must be 1 (grayscale) or 3 (RGB)")
    
    return image

def apply_convolution(image, kernel):
    """
    Apply convolution filter to an image
    Args:
        image (numpy.ndarray): Input image
        kernel (numpy.ndarray): Convolution kernel
    Returns:
        numpy.ndarray: Filtered image
    """
    # Assertions for input validation
    assert isinstance(image, np.ndarray), "Image must be a NumPy array"
    assert isinstance(kernel, np.ndarray), "Kernel must be a NumPy array"
    assert len(kernel.shape) == 2, "Kernel must be a 2D matrix"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel must have odd dimensions"
    assert len(image.shape) in [2, 3], "Image must be grayscale (2D) or RGB (3D)"
    
    if len(image.shape) == 3:  # RGB image
        assert image.shape[2] == 3, "RGB image must have 3 channels"
        height, width, channels = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        
        # Apply filter to each channel
        for c in range(channels):
            output[:, :, c] = convolve_channel(image[:, :, c], kernel)
            
        # Additional assertion for output size
        assert output.shape == image.shape, "Output size must match input size for RGB"
    else:  # Grayscale image
        output = convolve_channel(image, kernel)
        # Additional assertion for output size
        assert output.shape == image.shape, "Output size must match input size for grayscale"
    
    # Normalize to avoid values outside [0, 255] range
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def convolve_channel(image, kernel):
    """
    Apply convolution to a single channel
    Args:
        image (numpy.ndarray): Single channel image
        kernel (numpy.ndarray): Convolution kernel
    Returns:
        numpy.ndarray: Convolved channel
    """
    # Assertions for dimension validation
    assert image.shape[0] >= kernel.shape[0], "Image is too small for kernel in height"
    assert image.shape[1] >= kernel.shape[1], "Image is too small for kernel in width"
    
    # Image and kernel dimensions
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    
    # Add padding to handle borders
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)
    
    # Apply convolution
    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    # Additional assertion for output dimensions
    assert output.shape == image.shape, "Output channel size must match input channel size"
    
    return output

def display_images(images_dict, title="Image Comparison", figsize=(20, 15)):
    """
    Display multiple images in a single figure with captions
    Args:
        images_dict (dict): Dictionary with image names as keys and images as values
        title (str): Main title for the figure
        figsize (tuple): Figure size
    """
    num_images = len(images_dict)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, (name, image) in enumerate(images_dict.items()):
        plt.subplot(rows, cols, idx + 1)
        plt.title(name, fontsize=12)
        
        if len(image.shape) == 2:  # Grayscale
            plt.imshow(image, cmap='gray')
        else:  # RGB
            plt.imshow(image)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_kernels():
    """
    Create various convolution kernels
    Returns:
        dict: Dictionary of kernels
    """
    kernels = {}
    
    # 1. Blur kernel (average)
    kernels['blur_3x3'] = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
    
    # 2. Sobel horizontal (horizontal edge detection)
    kernels['sobel_horizontal'] = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    # 3. Sobel vertical (vertical edge detection)
    kernels['sobel_vertical'] = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    # 4. Sharpening kernel
    kernels['sharpen'] = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    # 5. Edge detection (Laplacian)
    kernels['laplacian'] = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    
    # 6. Emboss
    kernels['emboss'] = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])
    
    # 7. Gaussian blur 3x3
    kernels['gaussian_3x3'] = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ])
    
    # 8. Box blur 5x5
    kernels['blur_5x5'] = np.ones((5, 5)) / 25
    
    # 9. Box blur 7x7
    kernels['blur_7x7'] = np.ones((7, 7)) / 49
    
    # 10. Gaussian blur 5x5
    kernels['gaussian_5x5'] = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]) / 256
    
    # 11. Prewitt horizontal
    kernels['prewitt_horizontal'] = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    
    # 12. Prewitt vertical
    kernels['prewitt_vertical'] = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    
    # 13. Random kernels with different sizes
    kernels['random_3x3'] = np.random.uniform(-1, 1, (3, 3))
    kernels['random_5x5'] = np.random.uniform(-1, 1, (5, 5))
    kernels['random_7x7'] = np.random.uniform(-1, 1, (7, 7))
    
    # 14. High-pass filter
    kernels['high_pass'] = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    
    # 15. Motion blur
    kernels['motion_blur'] = np.array([
        [1/5, 0, 0, 0, 0],
        [0, 1/5, 0, 0, 0],
        [0, 0, 1/5, 0, 0],
        [0, 0, 0, 1/5, 0],
        [0, 0, 0, 0, 1/5]
    ])
    
    return kernels

def download_test_image():
    """Download a test image if not available locally"""
    image_url = "https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png"
    image_path = "test_image.png"
    
    if not os.path.exists(image_path):
        try:
            urlretrieve(image_url, image_path)
            print(f"Downloaded test image: {image_path}")
        except:
            print("Could not download test image. Please provide a local image.")
            return None
    
    return image_path

def save_results(images_dict, prefix=""):
    """
    Save filtered images to disk
    Args:
        images_dict (dict): Dictionary of images to save
        prefix (str): Prefix for filenames
    """
    for name, image in images_dict.items():
        filename = f"{prefix}_{name}.jpg" if prefix else f"{name}.jpg"
        
        if len(image.shape) == 3:  # RGB image
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, image_bgr)
        else:  # Grayscale image
            cv2.imwrite(filename, image)
        
        print(f"Saved: {filename}")

def main():
    """Main function to execute the convolution filters application"""
    
    # Download or use existing test image
    image_path = download_test_image()
    if image_path is None:
        print("Please provide a valid image path")
        return
    
    # Load images
    try:
        gray_image = image_load(image_path, 1)  # Grayscale
        rgb_image = image_load(image_path, 3)   # RGB
        print(f"Loaded images successfully")
        print(f"Grayscale image shape: {gray_image.shape}")
        print(f"RGB image shape: {rgb_image.shape}")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Create all kernels
    kernels = create_kernels()
    print(f"Created {len(kernels)} different kernels")
    
    # Apply filters to grayscale image
    gray_results = {'Original': gray_image}
    for kernel_name, kernel in kernels.items():
        try:
            filtered = apply_convolution(gray_image, kernel)
            gray_results[kernel_name] = filtered
            print(f"Applied {kernel_name} filter to grayscale image")
        except Exception as e:
            print(f"Error applying {kernel_name} to grayscale: {e}")
    
    # Apply filters to RGB image
    rgb_results = {'Original': rgb_image}
    for kernel_name, kernel in kernels.items():
        try:
            filtered = apply_convolution(rgb_image, kernel)
            rgb_results[kernel_name] = filtered
            print(f"Applied {kernel_name} filter to RGB image")
        except Exception as e:
            print(f"Error applying {kernel_name} to RGB: {e}")
    
    # Display results
    display_images(gray_results, "Grayscale Image Filtering Results")
    display_images(rgb_results, "RGB Image Filtering Results")
    
    # Save results
    save_results(gray_results, "gray")
    save_results(rgb_results, "rgb")
    
    # Display kernel information
    print("\n=== Kernel Information ===")
    for name, kernel in kernels.items():
        print(f"{name}: {kernel.shape}")
        if kernel.shape[0] <= 3:  # Only print small kernels
            print(f"  Values:\n{kernel}")
        print()

if __name__ == "__main__":
    main()