# Lettuce-Phenotyping-Pipeline
This repository contains the code and data used for extracting traits, capturing images, and analyzing the growth of lettuce cultivars (e.g., Rex and Rouxai). 

## Overview
`Trait_Extraction/` contains scripts to process image datasets (RGB and depth pairs) and extract key plant traits, including height, area, and volume, exporting results in CSV format.

`Growth_Curves/` contains code to compute and analyze growth curves for 20 REX and 20 ROUXAI samples based on height, area, and volume metrics, along with diurnal variation plots and detailed growth pattern analyses.

`Fresh_Weight_Analysis/` contains scripts to analyze fresh weight data using machine learning models, generating scatter plots of predicted vs observed values for various algorithms.

`Sample_Images/` contains example image datasets for testing the trait extraction pipeline.

## System Requirements
### Operating System
This repository is platform-independent and has been tested on Windows operating systems.
### Hardware Requirements
The scripts can run on standard computer systems. GPU support can be used via PyTorch for enhanced performance in certain computations.
### Package Requirements
The repository requires the following Python packages:
* `python 3.9.7`
* `pandas 2.0.0`
* `pytorch 2.0.0`
* `numpy 1.24.2`
* `matplotlib 3.7.2`
* `scikit-learn 1.3.0`

## Demo
Detailed examples for using the provided scripts are included:
* **Trait Extraction Pipeline**:
  * Run `trait_extraction.py` to process image datasets and extract lettuce widths, maximum depth and minimum depth value.
    *  The `trait_extraction.py` script requires the SAM model of type `vit_h`.
    * You can download it from the following link: [SAM Model (vit_h)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
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
2) Run the relevant scripts to process the data or analyze traits.
3) Review the generated outputs (e.g., CSV files, plots) in the designated folders or in the jupyter notebook. 

**Note:** Ensure proper configuration of the environment as per system and package requirements to avoid errors.

## Citation
`@article{,
author = {Your Name and Collaborators},
title = {Computer vision and IoT based plant phenotyping and growth monitoring with 3D Point Clouds},
journal = {TBD},
volume = {},
number = {},
pages = {},
doi = {},
abstract = {}
}`


 

 




