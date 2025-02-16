# FLIMngo

## Overview
We present **FLIMngo**, a novel network for predicting fluorescence lifetimes from raw TCSPC-FLIM data. 
Our model is based on the YOLOv5 architecture, which has been adapted for pixel-wise regression tasks. 


![yolo_git](https://github.com/user-attachments/assets/4fe74fbd-726d-4dbc-bf02-75955c33fc2c)


> **Deep learning for fluorescence lifetime predictions enables high-throughput in vivo imaging**          
> Sofia Kapsiani, Nino F Läubli, Edward N. Ward, Ana Fernandez-Villegas, Bismoy Mazumder, Clemens F. Kaminski, Gabriele S. Kaminski Schierle    
> <a href="https://www.ceb-mng.org/" target="_blank">Molecular Neuroscience Group</a> and <a href="https://laser.ceb.cam.ac.uk/" target="_blank">Laser Analytics Group</a> (University of Cambridge)
>
[[`bioRxiv`](https://www.biorxiv.org/content/10.1101/2024.09.13.612802v1)]  [[`bibtex`](#bibtex-citation)]


## Usage  

To make predictions using **FLIMngo**, the following parameters must be specified:  

- **Bin Width (ns)**: `bin_width` of time channels in nanoseconds for the raw data.  
- ***x*, *y* dimensions**: input data must have equal `x` and `y` dimensions (e.g., `256 × 256`).  
- **Time dimensions**:model currently only accepts raw data with **256 time dimensions**.  
  - for data that do not match this requirement, refer to `predict_diff_time_dimensions.ipynb` in `demo_notebooks` for a method to artificially expand/compress time dimensions.  
- **Normalisation**: time dimensions must be normalised to a range between `0` and `1`.  
  - See `demo_notebooks` for preprocessing steps.  
- **Background Masking**: the background should be masked either by intensity thresholding or by providing manual intensity masks (refer to `predict_celegans.ipynb` in `demo_notebooks` for details)  

Predictions can be made using the **pretrained file** named `flimngo_pretrained_v13102024.pth`.
Please note the model has been optimised for data collected with **IRFs** ranging from `100-400` ps.

## Demo
![test_git](https://github.com/user-attachments/assets/df51ff95-0a20-4ce8-8e71-b78983c7f7fd)

