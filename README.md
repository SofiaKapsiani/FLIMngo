# FLIMngo

## Overview
We present **FLIMngo**, a novel network for predicting fluorescence lifetimes from raw TCSPC-FLIM data. 
Our model is based on the YOLOv5 architecture, which has been adapted for pixel-wise regression tasks. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/ffc3970d-05bd-4c24-8a81-3a42a4be55a1" alt="icon_filimpa" width="800" height="410">
</div>


> **Deep learning for fluorescence lifetime predictions enables high-throughput in vivo imaging**          
> Sofia Kapsiani, Nino F Läubli, Edward N. Ward, Ana Fernandez-Villegas, Bismoy Mazumder, Clemens F. Kaminski, Gabriele S. Kaminski Schierle    
> <a href="https://www.ceb-mng.org/" target="_blank">Molecular Neuroscience Group</a> and <a href="https://laser.ceb.cam.ac.uk/" target="_blank">Laser Analytics Group</a> (University of Cambridge)
>
[[`bioRxiv`](https://www.biorxiv.org/content/10.1101/2024.09.13.612802v1)]  [[`bibtex`](#bibtex-citation)]


## Usage  

To make predictions using **FLIMngo**, the following parameters must be specified:  

- **Bin Width (ns)**: users must provide the `bin_width` of time channels in nanoseconds for the raw data.  
- ***x*, *y* dimensions**: input data must have equal `x` and `y` dimensions (e.g., `256 × 256`).  
- **Time dimensions**: the model currently only accepts raw data with **256 time dimensions**.  
  - if your data do not match this requirement, refer to `predict_diff_time_dimensions.ipynb` in `demo_notebooks` for a method to artificially expand/compress time dimensions.  
- **Normalisation**: time dimensions must be normalised to a range between `0` and `1`.  
  - See `demo_notebooks` for preprocessing steps.  
- **Background Masking**: the background should be masked using either intensity thresholding or by providing manual intensity masks (refer to `predict_celegans.ipynb` in `demo_notebooks` for details)  
