# DeepDISC
Using deep learning for Detection, Instance Segmentation, and Classification on astronomical survey images.

*Reference Paper:* [Merz et al. 2023](https://academic.oup.com/mnras/article/526/1/1122/7273850)

*Corresponding Author:* 
[Grant Merz](gmerz3@illinois.edu), University of Illinois at Urbana-Champaign


This is an updated repo of the original implementation (https://github.com/burke86/astro_rcnn)

## Description:

DeepDISC is a deep learning framework for efficiently performing source detection, classification, and segmnetation (deblending) on astronomical images.  We have built the code using detectron2 https://detectron2.readthedocs.io/en/latest/ for a modular design an access to state-of-the-art models. 

## Installation:

1) Create a conda environment, with python>=3.10.  You can use the environment.yml file provided and run  
   `conda env create -f environment.yml`  

or create an environment from scratch and install by hand the packages listed in the environment.yml file  

2) Install deepdisc by cloning this repo and running
   `pip install [e].` 
   [e] is optional and will install in editable mode.  Use if you are going to change the source code.


Usage:
```
demo_btk.ipynb
```
This notebook uses simulated images generated using the Blending Toolkit (Mendoza et al 2025).  We used the CatSim (Connolly et al 2014) example catalog provided within the BTK to generate a set of training/test images, and then constructed per-image metadata the network needs to train. The notebook is largely for demo purposes, so does not include full training scheduling and optimizations.


The BTK simulated data can be downloaded [here](https://uofi.box.com/s/gqnlza5ldogumqbrqlbaloh3xtpjxzwo)


Mendoza, Ismael, Andrii Torchylo, Thomas Sainrat, Axel Guinot, Alexandre Boucaud, Maxime Paillasa, Camille Avestruz, et al. 2025. “The Blending ToolKit: A Simulation Framework for Evaluation of Galaxy Detection and Deblending.” The Open Journal of Astrophysics 8 (February). https:/​/​doi.org/​10.33232/​001c.129699.

A.J. Connolly et al., An end-to-end simulation framework for the Large Synoptic Survey
Telescope, in Proc. SPIE 9150 (2014) 14.
