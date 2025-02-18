# FL🦩Mngo




## Overview
We present **FLIMngo**, a novel network for predicting fluorescence lifetimes from raw TCSPC-FLIM data. 
Our model is based on the YOLOv5 architecture, which has been adapted for pixel-wise regression tasks. 



![yolo_git](https://github.com/user-attachments/assets/d2b4473c-cf28-4c3a-8a37-4a4d68f15ff0)


> **Deep learning for fluorescence lifetime predictions enables high-throughput in vivo imaging**          
> Sofia Kapsiani, Nino F Läubli, Edward N. Ward, Ana Fernandez-Villegas, Bismoy Mazumder, Clemens F. Kaminski, Gabriele S. Kaminski Schierle    
> <a href="https://www.ceb-mng.org/" target="_blank">Molecular Neuroscience Group</a> and <a href="https://laser.ceb.cam.ac.uk/" target="_blank">Laser Analytics Group</a> (University of Cambridge)
>
[[`bioRxiv`](https://www.biorxiv.org/content/10.1101/2024.09.13.612802v1)]  [[`bibtex`](#bibtex-citation)]


## Usage 

```bash
git clone https://github.com/SofiaKapsiani/FLIMngo.git
cd FLIMngo

# Create and activate a Conda environment
conda create --name flimngo_env python=3.9 -y
conda activate flimngo_env

# Install dependencies
pip install -r requirements.txt
```

Predictions can be made using the **pretrained model** file, `flimngo_pretrained_v13102024.pth`.

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

- **`predict_simulated.ipynb`**: Evaluates performance on synthetic FLIM data with varying photon counts per pixel.  
- **`predict_reduced_photon_counts.ipynb`**: Demonstrates performance on images from different experiments with at least **100 photon counts per pixel**, as well as the same images with artificially reduced photon counts (**10–100 photons per pixel**).  
- **`predict_diff_time_dimensions.ipynb`**: Example of predicting lifetimes from input data that do not have **256 time dimensions**, with a method for time dimension adjustment.  
- **`predict_celegans_dynamic.ipynb`**: Predicting lifetimes from dynamic, non-anesthetised *C. elegans*.

## Data simualtion

![methodology_v2_git](https://github.com/user-attachments/assets/c58090e8-152d-4b79-98d1-d35bb081602b) 

The fluorescence intensity images shown in **(a)** are taken from the **Human Protein Atlas (HPA) dataset** ([Kaggle HPA Single-Cell Image Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification)).  

HPA images consist of **RGBY** color channels, representing:  
- **R** (Red) – Microtubules  
- **G** (Green) – Protein  
- **B** (Blue) – Nucleus  
- **Y** (Yellow) – Endoplasmic Reticulum (ER)  

#### **Execution Order**  

To generate simulated FLIM data, run the notebooks found in `data_simulation` directory in the following order:  

1. **`notebook1_cropp_imgs.ipynb`**  
   - Applies a sliding window approach to extract **256×256 pixel sub-images** (x, y) from the HPA fluorescence intensity images, as shown in **(a)**.  

2. **`notebook2_irf_simulation.ipynb`**  
   - Generates a dataset containing both **experimentally acquired** and **simulated instrument response functions (IRFs)**.  

3. **`notebook3_lifetime_simulation.ipynb`**  
   - Simulates **3D FLIM data** by assigning a fluorescence lifetime range to each HPA color channel, as illustrated in **(b)**.  
   - **Perlin noise** (example in **(c)**) is used to determine the fractional contribution of the first color channel to each pixel.  
   - For each **pixel**, fluorescence decay curves are simulated using the following equation:  

   
   $$
   y(t) = IRF \otimes \sum_{i=1}^{n} \left( a_i e^{-t/\tau_i} \right) + \text{noise} \tag{1}
   $$  

   where:  
  -  $$IRF$$ represents the instrument response function.  
  - $$n$$ is the number of lifetime components (i.e., the number of color channels contributing to the pixel).  
  - $$a_i$$ and $$\tau_i$$ are the fractional contribution and fluorescence lifetime of each color channel at a given pixel, respectively.  
  - $$\text{noise}$$ accounts for the Poisson noise typically encountered in TCSPC systems.  
  - $$\otimes$$ denotes the convolution between the decay curve and the IRF.  

For further details, please refer to the **Methods** section of our manuscript. 


# Citation

*If you found **FLIMngo** helpful, please consider citing our work!* 😊
<a name="bibtex-citation"></a>
```

```


