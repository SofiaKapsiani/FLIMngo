import os
import numpy as np
import torch
import sdtfile as sdt

def load_sdt(data_dir, file_name, normalise = True):
  """
    Function for loading Becker & Hickl '.sdt' files.

    Parameters:
    - data_dir (str): Directory where the file is located.
    - file_name (str): Name of the '.sdt' file to load.
    - normalise (bool): If True, normalides the data along the time axis

    Returns:
    - torch.Tensor: The loaded data as a PyTorch tensor, optionally normalised.
    """

  file_path = os.path.join(data_dir, file_name)
  
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_name} not found.")
    
  sdt_file = sdt.SdtFile(file_path)
  data = np.moveaxis(sdt_file.data[0], -1, 0)
  data = data.astype(np.float32)

  if normalise == True:
    max_values = np.nanmax(data, axis=0)  # Use np.nanmax to ignore NaNs
    data_norm = data / max_values  # Element-wise division
    data_norm = np.nan_to_num(data_norm, nan=0.0)  # Replace NaNs with 0
    data_tensor = torch.tensor(data_norm)
  else:
    data_tensor = torch.tensor(data)
  
  return data_tensor