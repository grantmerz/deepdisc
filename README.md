# DeepDISC
Using deep learning for Detection, Instance Segmentation, and Classification on astronomical survey images.

*Reference Paper:* [Merz et al. 2023](https://academic.oup.com/mnras/article/526/1/1122/7273850)

*Corresponding Author:* 
[Grant Merz](gmerz3@illinois.edu), University of Illinois at Urbana-Champaign


This is an updated repo of the original implementation (https://github.com/burke86/astro_rcnn)

## Description:

DeepDISC is a deep learning framework for efficiently performing source detection, classification, and segmnetation (deblending) on astronomical images.  We have built the code using detectron2 https://detectron2.readthedocs.io/en/latest/ for a modular design an access to state-of-the-art models. 

## Installation:

1) Create a conda environment.  We recommend using python 3.9.  You can use the environment.yml file provided and run  
   `conda env create -f environment.yml`  

or create an environment from scratch and install by hand the packages listed in the environment.yml file  

2) Install deepdisc with  
   `pip install deepdisc`  
   You can also install by cloning this repo and running `pip install [e].`  [e] is optional and will install in editable mode.  Use if you are going to change the source code  

Usage:
```
demo_hsc.ipynb
```
This notebook follows a very similar procedure to ```demo_decam.ipynb```, but for real HSC data.  The ground truth object locations and masks are constructed following ```training_data.ipynb``` and classes are constructed with external catalog matching following ```hsc_class_assign.ipynb``` It is largely for demo purposes, so will not reflect the results of the paper.  The training scripts we used to recreate the paper results are in ```train_decam.py``` and ```train_hsc_primary.py```  


