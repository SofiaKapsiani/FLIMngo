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
Predictions can be made using the **pretrained model** file, `flimngo_pretrained_v13102024.pth`.

```bash
git clone https://github.com/SofiaKapsiani/FLIMngo.git
cd FLIMngo

# Create and activate a Conda environment
conda create --name flimngo_env python=3.9 -y
conda activate flimngo_env

# Install dependencies
pip install -r requirements.txt
```

### Parameters

- **Bin Width (ns)**: `bin_width` of time channels in nanoseconds for the raw data.  
- **X, Y Dimensions**: Input data must have equal `x` and `y` dimensions (e.g., `256 × 256`).  
- **Time Dimensions**: The model currently only accepts raw data with **256 time dimensions**.  
  - For data that do not match this requirement, refer to `predict_diff_time_dimensions.ipynb` in `demo_notebooks` for a method to artificially expand/compress time dimensions.  

### Preprocessing  

- **Normalisation**: Time dimensions should be normalised to a range between `0` and `1`.  
  - See preprocessing steps in  `demo_notebooks`.
- **Background Masking**: The background should be masked using either:  
  - *Intensity thresholding*  
  - *Manual intensity masks* (refer to `predict_celegans.ipynb` in `demo_notebooks` for details).  

Please note the model has been optimised for data collected with **IRFs** ranging from `100-400` ps.

## Demo

FLIMngo maintains high prediction accuracy even for FLIM data with fluorescence decay curves containing as few as 10 photon counts.

![test_git](https://github.com/user-attachments/assets/df51ff95-0a20-4ce8-8e71-b78983c7f7fd)

### Notebooks  

- **`predict_simulated.ipynb`**: Demonstrates FLIMngo's performance on synthetic FLIM data, where the same image is simulated with varying photon counts per pixel.  
- **`predict_reduced_photon_counts.ipynb`**: Provides examples of images acquired from different experiments with at least **100 photon counts per pixel**, alongside the same images with artificially reduced photon counts (**10–100 photons per pixel**).  


