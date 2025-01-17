# Lettuce-Phenotyping-Pipeline
This repository contains the code and data used for extracting traits, capturing images, and analyzing the growth of lettuce cultivars (e.g., Rex and Rouxai). 

## Overview
`Data_capture/` contains script, which is used to interface with imaging hardware and capture synchronized RGB and depth images of lettuce samples. 

`Trait_Extraction/` contains scripts to process image datasets (RGB and depth pairs) and extract key plant traits, including height, area, and volume, exporting results in CSV format.

`Growth_Curves/` contains code to compute and analyze growth curves for 20 REX and 20 ROUXAI samples based on height, area, and volume metrics, along with diurnal variation plots and detailed growth pattern analyses.

`Fresh_Weight_Analysis/` contains scripts to analyze fresh weight data using machine learning models, generating scatter plots of predicted vs observed values for various algorithms.


## System Requirements
### Operating System
This repository is platform-independent and has been tested on Windows operating systems.
### Hardware Requirements
The scripts can run on standard computer systems. GPU support can be used via PyTorch for enhanced performance in certain computations.
### Package Requirements
The repository requires the following Python packages:
* `python`
* `pandas 2.1.4`
* `torch 2.2.0`
* `numpy 1.24.2`
* `matplotlib 3.8.2`
* `scikit-learn 1.3.2`
* `scipy 1.12.0`
* `seaborn 0.13.2`
* `ultralytics 8.3.59`
* `imageio 2.33.1`
* `opencv-python 4.9.0.80`

## Demo
Detailed examples for using the provided scripts are included:
* **Collecting Images**:
  * Use `data_capture.py` (located in the `Data_capture` folder) to collect RGB and Depth images of lettuce samples. Captured images are automatically saved with filenames including the date and time of capture (e.g., `image_2024-03-27 14_19_56.png`).
 
**Note: To run this script, you need an imaging hardware system comprised of an Intel RealSense D435 camera integrated with a Raspberry Pi. The script should be executed on this system to collect images. If you wish to create this system, please follow the steps mentioned in this [link](https://github.com/datasith/Ai_Demos_RPi/wiki/Raspberry-Pi-4-and-Intel-RealSense-D435). Alternatively, sample data is available in the `/Trait_Extraction/Sample_Images` folder, and the complete dataset used in this project is publicly available on Zenodo: [RGBD_DATA](https://zenodo.org/records/14635169?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZlZDgyZjE4LTA4ZWYtNGY5ZS1hYTc1LWM4ODI2YWVhM2RhYiIsImRhdGEiOnt9LCJyYW5kb20iOiJlZTJkZDBlOGFlYTY1YTZlODFmMTczZDNhMmI4Y2UyYiJ9.SCAFf-zoOMV9RpX4HbqEUFhDFY6GCuzOT72SycuOGTuHe-rPUdZ7ZCDELwNmKsAb_xi3Dx50OeGYsqtbtpHgmQ)**.

* **Trait Extraction Pipeline**:
  * Run `trait_extraction_SAM.py/trait_extraction_FastSAM.py` to process image datasets and extract lettuce widths, maximum depth and minimum depth value.
    * The `trait_extraction_SAM.py` script requires the SAM model of type `vit_h`. You can download it from the following link: [SAM Model (vit_h)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
    * The `trait_extraction_FastSAM.py` script requires the FastSAM model of type `FastSAM-s`. You can download it from the following link: [FastSAM Model (FastSAM-s)](https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt).
    * Make sure the model file is placed in the appropriate directory and correctly referenced in the script.
  * Use `compute_traits.ipynb` to compute height, area, and volume traits and save the output to a CSV file.
* **Growth Curves Analysis**:
  * Run `growth_curves.ipynb` to visualize growth curves of the 20 REX and ROUXAI samples over time.
  * Use `diurnal.ipynb` for visualizing diurnal variations in traits.
  * Use `growth_patterns_analysis.ipynb` for advanced consistency analysis
* **Fresh Weight Analysis**:
  * Run `fresh_weight_analysis.ipynb` to generate scatter plots for different machine learning models:
    * `Decision Tree Model`
    * `Random Forest Model`
    * `Gradient Boosting Regressor Model`
    * `Gaussian Process Regressor Model`

## Instructions for Use
1) Place your dataset in the appropriate directory (e.g. `⁣Sample Images/`).
2) Run the relevant scripts to process the image data or analyze traits.
3) Review the generated outputs (e.g., CSV files, plots) in the designated folders or in the jupyter notebook. 

**Note:** Ensure proper configuration of the environment as per system and package requirements to avoid errors.

## Citation
`@article{, author = {Akash Ajagekar, Yu Jiang, Fengqi You},
title = {Computer vision and IoT based plant phenotyping and growth monitoring with 3D Point Clouds},
journal = {TBD},
volume = {},
number = {},
pages = {},
doi = {},
abstract = {}
}`
