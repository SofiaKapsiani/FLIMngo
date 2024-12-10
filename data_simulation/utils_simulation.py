import os
import numpy as np
import torch
from PIL import Image
import sdtfile as sdt
from skimage import transform

# --- cropp_imgs.ipynb --- 
def colour_channel_imgs(img_dir, img):
    '''Load and open each colour channel image
    red: microtubules,
    green: prot. of interest
    blue: nucleus
    yellow: ER'''
    img_path = os.path.join(img_dir, img)
    return [Image.open(img_path+'_red.png'), Image.open(img_path+'_green.png'),
                Image.open(img_path+'_blue.png'), Image.open(img_path+'_yellow.png')]

# --- irf_simulation.ipynb --- 
def load_sdt(data_dir, file_name):
    """ Read Becker & Hickl .sdt files"""
    sdt_file = sdt.SdtFile(os.path.join(data_dir, file_name))
    data = np.moveaxis(sdt_file.data[0], -1, 0).astype(np.float32)
    t_series = sdt_file.times[0].astype(np.float32)

    return data, t_series*10**9

def normalise_irf(irf):
    """ Normalise IRF to have values between 0 and 1"""
    irf = irf.flatten()
    irf_norm = irf/irf.max()
    return irf_norm

def resample_array(arr, new_size):
    """ Reduces dimentions of an array to match desired dimentions 
        by averaging values from the original array"""
    # calculate the reduction factor 
    factor = len(arr) / new_size
    # initalise new array to save data
    reduced_array = np.zeros(new_size)
    # loop through new array
    # calculate the range of indices in the original array that correspond to the current index in new array
    for i in range(new_size):
        start_index = int(i * factor)
        end_index = int((i + 1) * factor)
        reduced_array[i] = np.mean(arr[start_index:end_index])
    return reduced_array

# --- lifetime_simulation.ipynb ---

def getPerlin(X,Y,octaves=5,persistence=0.6):
    """
    Function to generate perlin noise
    Args:
        X: The x dimension of the perlin noise array
        Y: The y dimension of the perlin noise array
        octaves: The number of octaves to be used in the perlin noise generation
        persistence: The persistence to be used in the perlin noise generation

    Returns:
        perlin: The perlin noise array
    """
    perlin = np.zeros((X,Y,octaves)) # Initialize the numpy array

    for i in range(octaves):
        # Generate an x BY y array of random numbers between 0 and 1
        x = int(X/2**i) # x = (np.floor()X/n)
        y = int(Y/2**i) # x = (np.floor()X/n)
        Z = np.random.rand(x,y)
        # Upsample the array to X by Y
        perlin[:,:,i] = transform.resize(Z, (X,Y))
        perlin[:,:,i] = perlin[:,:,i]*persistence**(octaves-i)
    # Plot the original and upsampled arrays

    perlin = np.sum(perlin,2) # Sum the octaves
    perlin = perlin-np.amin(perlin) # Normalize the data to 0 and 1
    perlin = perlin/np.amax(perlin)

    return perlin


def tau_channel(img, dim =[256, 256], tau_range = [0.1, 10.0], tau_dist = [0.2, 2]):
    '''Generates the range of tau observed in a single colour channel'''
    tau = torch.tensor(np.random.uniform(tau_range[0], tau_range[1]))  # randomly assign a tau value for each colour channel
    diff_range = torch.tensor(np.random.uniform(tau_dist[0], tau_dist[1]))  # randomly select the difference between the min and max tau observed in the same colour channel
    tau_matrix = torch.rand(dim[0], dim[1]) * diff_range + tau  # matrix of 128 x 128 dim with random tau distribution
    tau_img = torch.where(img == 0, torch.tensor(0.0), tau_matrix)  # assign tau values on non-background pixels of microscopy image
    return tau_img


def calculate_ratio(random_value, num_parameters):
    '''Generates tau contribultion of each colour channel to the final lifetime image'''
    ratio = []
    remaining_ratio = 1.0

    for i in range(num_parameters - 1):
        contribution = random_value * remaining_ratio
        ratio.append(contribution)
        remaining_ratio -= contribution

    ratio.append(remaining_ratio)
    return ratio
