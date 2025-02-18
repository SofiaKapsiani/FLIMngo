## Data simualtion

![methodology_v2_git](https://github.com/user-attachments/assets/c58090e8-152d-4b79-98d1-d35bb081602b) 

The fluorescence intensity images shown in **(a)** are taken from the **Human Protein Atlas (HPA) dataset** ([Kaggle HPA Single-Cell Image Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification)).  

HPA images consist of **RGBY** color channels, representing:  
- **R** (Red) – Microtubules  
- **G** (Green) – Protein  
- **B** (Blue) – Nucleus  
- **Y** (Yellow) – Endoplasmic Reticulum (ER)  

#### **Execution Order**  

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

