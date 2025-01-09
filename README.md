# Lettuce-Phenotyping-Pipeline
This repository contains the code and data used for extracting traits, capturing images, and analyzing the growth of lettuce cultivars (e.g., Rex and Rouxai). 

## Overview
`Trait Extraction Pipeline/` contains scripts to process image datasets (RGB and depth pairs) and extract key plant traits, including height, area, and volume, exporting results in CSV format.

`Growth Curves/` contains code to compute and analyze growth curves for 20 REX and 20 ROUXAI samples based on height, area, and volume metrics, along with diurnal variation plots and detailed growth pattern analyses.

`Fresh Weight Analysis/` contains scripts to analyze fresh weight data using machine learning models, generating scatter plots of predicted vs observed values for various algorithms.

`Sample Images/` contains example image datasets for testing the trait extraction pipeline.

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
  * Use `compute_traits.py` to compute height, area, and volume traits and save the output to a CSV file.
* **Growth Curves Analysis**:
  * Run `growth_curves.py` to visualize growth curves of the 20 REX and ROUXAI samples over time.
  * Use `diurnal.py` for visualizing diurnal variations in traits.
  * Use `growth_patterns_analysis.py` for advanced analysis, including:
    * Individual growth patterns with mean overlay.
    * Histogram of Pearson correlation coefficients.
    * Boxplot of differences from the mean pattern.
    * Line plot of coefficient of variation across time periods.
* **Fresh Weight Analysis**:
  * Run `fresh_weight_analysis.py` to generate scatter plots for different machine learning models:
    * `Decision Tree Model`
    * `Random Forest Model`
    * `Gradient Boosting Regressor Model`
    * `Gaussian Process Regressor Model`

## Instructions for Use
1) Place your dataset in the appropriate directory (e.g.,⁣Sample Images/).
2) Run the relevant scripts to process the data or analyze traits.
3) Review the generated outputs (e.g., CSV files, plots) in the designated folders

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


 

 




