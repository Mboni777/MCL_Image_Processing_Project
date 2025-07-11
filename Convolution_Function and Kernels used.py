# Alongside the given kernels (transformation functions), we added a fifth one (Gaussian) to explore the image more deeply and reveal extra details about it.
# The following are different convolution kernels we will use, where each give different effect

# Blur Kernel
blur_kernel = 1/9 * np.array( [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

#  sharpen kernel
sharpen_kernel = sharpen_kernel = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]])

# edge detection kernel
edge_kernel = np.array([ [-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])

# Emboss Kernel
emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

# Gausian blur kernel (Added by our team to explore more and uncover more about the image)
gaussian_kernel = 1/16 * np.array([[1, 2, 1],[2, 4, 2], [1, 2, 1]])



def convolution_function(image, kernel, mode="display"):
    """
  This function applies a convolution filter to a 2D image using the kernel provided.
  If mode is "display", it prepares the output as a normal image that can be shown using imshow. and if  mode is "raw", it keeps the exact values for calculations like gradients.
    """
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Pad the image to preserve dimensions
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    # Create an empty output image
    output_image = np.zeros_like(image, dtype=float)

    # Apply convolution
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output_image[i, j] = np.sum(region * kernel)

    # Handle output type
    if mode == "display":
        output_image = np.clip(output_image, 0, 255)
        return output_image.astype(np.uint8)
    else:
        return output_image
