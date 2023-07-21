# Probability-Based-Boundary-Detection

This repository consists of comparison of baseline edge detection algorithms like Canny and Sobel with [Probability of boundary detection algorithm](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf). A simpler version of PB algorithm has been implemented which considers texture, color and intensity discontinuities. This algorithm predicts per pixel probability of the boundary detected. The original image and the output of implementation is shown below:

<img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/a857633f9c805d2ab404d2ee422ae9dac2f94aba/Outputs/10.jpg" align="center" alt="Original" width="400"/> <img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/a857633f9c805d2ab404d2ee422ae9dac2f94aba/Outputs/Using%20all%20Filters/IMAGE%2010/PbLite10.png" alt="PBLite" width="400"/>

The algorithm of PBLite detection is shown below:

<img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/da6f2ff26d046948fb67f22ea9debe75c1f68a7d/Outputs/hw0.png" align="center" alt="PBLite"/>

The main steps for implementing the same are:

## Step 1: Feature extraction using Filtering
The filter banks implemented for low-level feature extraction are Oriented Derivative if Gaussian Filters, Leung-Malik Filters (multi-scale) and Gabor Filter.

<img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/b3b2e0412f7ded1795bfcfc32f11a4117487ef9f/Outputs/Using%20all%20Filters/DoG.png" align="center" alt="DoG" width="250"/> <img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/b3b2e0412f7ded1795bfcfc32f11a4117487ef9f/Outputs/Using%20all%20Filters/LM.png" align="center" alt="PBLite" width="250"/><img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/b3b2e0412f7ded1795bfcfc32f11a4117487ef9f/Outputs/Using%20all%20Filters/GB" align="center" alt="PBLite" width="250"/>

## Step 2: Extracting texture, color and brightness using clustering
Filter banks can be used for extraction of texture properties but here all the three filter banks are combined which results into vector of filter responses. As filter response vectors are generated, they are clustered together using k-means clustering. For Texton Maps k = 64 is used; Color and Brightness Maps k= 16 is used.


<img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/eb0ab5bede8fa18b666c3d3bf1886afc6353da77/Outputs/Using%20all%20Filters/IMAGE%2010/TextonMap_10.png" align="center" alt="DoG" width="250"/> <img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/eb0ab5bede8fa18b666c3d3bf1886afc6353da77/Outputs/Using%20all%20Filters/IMAGE%2010/C_Map_10.png" align="center" alt="PBLite" width="250"/> <img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/eb0ab5bede8fa18b666c3d3bf1886afc6353da77/Outputs/Using%20all%20Filters/IMAGE%2010/B_Map_10.png" align="center" alt="PBLite" width="250"/>

The gradient measurement is performed to know how much all features distribution is changing at a given pixel. For this purpose, half-disc masks are used.

<img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/eb0ab5bede8fa18b666c3d3bf1886afc6353da77/Outputs/Using%20all%20Filters/IMAGE%2010/Tg_10.png" align="center" alt="PBLite" width="250"/> <img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/eb0ab5bede8fa18b666c3d3bf1886afc6353da77/Outputs/Using%20all%20Filters/IMAGE%2010/Cg_10.png" align="center" alt="PBLite" width="250"/> <img src="https://github.com/ParamDave5/Probability-Based-Boundary-Detection/blob/eb0ab5bede8fa18b666c3d3bf1886afc6353da77/Outputs/Using%20all%20Filters/IMAGE%2010/Bg_10.png" align="center" alt="PBLite" width="250"/>

## Step 3: Pb-Score
The gradient maps which are generated are combined with classical edge detectors like Canny and Sobel baselines for weighted avaerage.

## Run Instructions
```
python Wrapper.py
```
# File structure
    Phase1
    ├── Code
    |  ├── Wrapper.py
    ├── Data
    |  ├── BSDS500
    ├── results
    |  ├── BrightnessGradient
    |  ├── Brightness_map
    |  ├── ColorGradient
    |  ├── Color_map
    |  ├── PbLite
    |  ├── TextonGradient
    |  ├── TextonMap
    |  ├── Other filter outputs

This was implemented as part of [CMSC733](https://cmsc733.github.io/2022/hw/hw0/) and for detailed report refer [here](https://github.com/naitri/Probability_based_Boundary_Detection/blob/main/Report.pdf)
