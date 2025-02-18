import os
import numpy as np
import math
import tifffile as tiff
import sdtfile as sdt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_tiff(dir, filename):
    """
    Loads a .tif or .tiff file and returns the data as a NumPy array.
    The data is expected to have dimensions: [time, x, y].
    Args:
        sir (str): The directory where the file is located.
        filename (str): The name of the file to be loaded.
    """
    try:
        file_path = os.path.join(dir, filename)
        raw_data = tiff.imread(file_path).astype('float32')
        return raw_data
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found in : {file_path}")

def load_sdt(dir, filename):
    """
    Loads an SDT file and returns the data with dimentions as time, x, y.
    Args:
        dir (str): Directory containing the SDT file.
        filename (str): Name of the SDT file.
    Returns: data and data bin width in nanoseconds (required from making pedictions)
    """
    try:
      # Load the SDT file
      file_path = os.path.join(dir, filename)
      sdt_file = sdt.SdtFile(file_path)
      # Process data: move time axis to the first position and convert to float32
      data = np.moveaxis(sdt_file.data[0], -1, 0).astype(np.float32)
      # Print time resolution (difference between consecutive time points)
      t_series = sdt_file.times[0].astype(np.float32)
      bin_width = (t_series[1] - t_series[0])*10**9 # gets bin width in nanosecond (needed for model preditions)
  
      return data, bin_width
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found in : {file_path}")


def normalise(data):
    """
    Normalises the input data along the time axis.
    Args: data (np.ndarray): Input array to normalize.
    """
    max_values = np.nanmax(data, axis=0)      # Maximum values ignoring NaNs
    max_values[max_values == 0] = 1           # Avoid division by zero
    data_norm = data / max_values             # Scale data by max values
    data_norm = np.nan_to_num(data_norm, nan=0.0)  # Replace NaNs with 0
    return data_norm

def mask_data(data, mask):
    """
    Apply manual masks to data.
    """
    masked_data = np.where(mask == 0, 0, data)
    return masked_data 

def mask_intensity(data, min_photons=100, max_photons=1000000):
    """
    Masks data based on intensity thresholds.
    Args:
        data (np.ndarray): Input 3D data array (time, height, width).
        min_photons (int): Minimum intensity threshold; values below this will be masked (set to 0).
        max_photons (int): Maximum intensity threshold; values above this will be masked (set to 0).
    """
    # Compute the intensity by summing along the time axis
    intensity = data.sum(0)

    # Mask data where intensity is below the minimum threshold
    masked_data = np.where(intensity < int(min_photons), 0, data)

    # Further mask data where intensity exceeds the maximum threshold
    masked_data = np.where(intensity > int(max_photons), 0, masked_data)

    # Return the masked data
    return masked_data

def visualise_images(images, filenames, cmap_c='jet', columns=3, vmin=None, vmax=None, save_path=None):
   
    # Number of columns for subplots (user-defined)
    cols = columns
    # Calculate the number of rows dynamically
    rows = math.ceil(len(images) / cols)

    # Create subplots with calculated rows and fixed columns
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2.6, rows * 2.5))  # Adjust figsize dynamically

    for i, ax in enumerate(axs.flat):
        if i < len(images):
            out = images[i]
            out[out == 0] = np.nan
            
            # Display the image
            img = ax.imshow(out, cmap=cmap_c, vmin=vmin, vmax=vmax)
            
            # Create an axis on the right side of the current axis for the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            # Add a colorbar with matching height
            plt.colorbar(img, cax=cax)
            
            # Set the title using the Axes object
            #mean_value = str(np.round(np.nanmean(out), 2))  # Calculate mean ignoring NaNs
            ax.set_title(f'{filenames[i]}')
            
            # Set background color and remove ticks
            ax.patch.set_facecolor((0, 0, 0, 1.0))
            ax.set_xticks([])  # Remove x ticks
            ax.set_yticks([])  # Remove y ticks
        else:
            # Hide the last subplot if there are no more images to display
            ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
  
    # save the figure if a save_path is provided
    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches="tight")
      print(f"Figure saved to {save_path}")
      
    # Show the plot
    plt.show()

# --- other --- #
def resize_data_tdim(data, time_dim_target=256):
    """
    Resizes the time dimension of 3D data to the target time dimention (i.e. 256) using interpolation.
    Args:
        data (np.ndarray): 3D array (time, height, width).
        time_dim_target (int): Target size for the time dimension (default: 256).
        
    """
    from scipy.ndimage import zoom

    # Calculate scaling factors (time axis is resized, others remain unchanged)
    zoom_factors = (time_dim_target / data.shape[0], 1, 1)

    # Interpolate data to match the target size
    return zoom(data, zoom_factors, order=1)  # Linear interpolation

def resize_time(data, bin_width):
    """Resize time axis, from data time dimention size and bin with. Target time axis dimentions is 256"""
    time = (np.arange(0, data.shape[0]) * bin_width)[:data.shape[0]]
    # Original indices and desired indices
    original_indices = np.linspace(0, 1, len(time))
    desired_indices = np.linspace(0, 1, 256)
    
    # Interpolating the data
    time_interpolated = np.interp(desired_indices, original_indices, time)
    return time_interpolated

