import os
import numpy as np
from PIL import Image
import sdtfile as sdt

# --- cropp_imgs.ipynb --- #
def colour_channel_imgs(img_dir, img):
    '''Load and open each colour channel image
    red: microtubules,
    green: prot. of interest
    blue: nucleus
    yellow: ER'''
    img_path = os.path.join(img_dir, img)
    return [Image.open(img_path+'_red.png'), Image.open(img_path+'_green.png'),
                Image.open(img_path+'_blue.png'), Image.open(img_path+'_yellow.png')]

# <--- irf_simulation.ipynb  -->
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


