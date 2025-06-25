""" This is a demo "solo config" file for use in the demo_hsc_swin notebook.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import numpy as np
import os
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs=1
metadata = OmegaConf.create() 
metadata.classes = ["galaxy"]

numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, model, train, lr_multiplier, optimizer
import deepdisc.model.loaders as loaders
from deepdisc.data_format.augment_image import train_augs
from deepdisc.data_format.image_readers import HSCImageReader
from deepdisc.model.shear_models import CNNShearROIHeads, RedshiftPDFCasROIHeads

# Overrides
dataloader.augs = train_augs
dataloader.train.total_batch_size = bs
dataloader.epoch = 50

# ---------------------------------------------------------------------------- #
# For the roiheads
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.proposal_generator.batch_size_per_image = 512

model.roi_heads._target_ = CNNShearROIHeads
#model.roi_heads.num_components = 3
#model.roi_heads.zloss_factor = 1
model.roi_heads.shear_factor = 1e1
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 600
model.roi_heads.positive_fraction = 0.33


# ---------------------------------------------------------------------------- #
#Change for different data sets

#This is the number of color channels in the images
model.backbone.bottom_up.in_chans = 6
model.pixel_mean = [0.06687771, 0.06493237, 0.09773403, 0.13505115, 0.17717863, 0.26844701]
model.pixel_std = [3.39803446, 2.23477371, 2.99209274, 4.05513938, 5.42771314, 8.59110257]

# ---------------------------------------------------------------------------- #
model.proposal_generator.nms_thresh = 0.3

for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 2000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.3

#The ImageNet1k pretrained weights file.  Update to your own path
train.init_checkpoint = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_246a82.pkl"

optimizer.lr = 1e-3
#dataloader.test.mapper = loaders.DictMapper
#dataloader.train.mapper = loaders.DictMapper
#dataloader.epoch=epoch


# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yacs_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
SOLVER.IMS_PER_BATCH = bs

DATASETS.TRAIN = "astro_train"
DATASETS.TEST = "astro_val"

SOLVER.BASE_LR = 0.001
SOLVER.CLIP_GRADIENTS.ENABLED = True
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0


SOLVER.STEPS=[]
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
TEST.DETECTIONS_PER_IMAGE = 3000
#SOLVER.MAX_ITER = efinal  # for DefaultTrainer
