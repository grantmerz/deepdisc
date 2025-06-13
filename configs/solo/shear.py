""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import numpy as np
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs = 24

metadata = OmegaConf.create() 
metadata.classes = ["galaxy", "star"]

numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, model, train, lr_multiplier, optimizer
import deepdisc.model.models as roiheads
import deepdisc.model.loaders as loaders
import deepdisc.model.meta_arch as meta_arch 
from deepdisc.data_format.augment_image import dc2_train_augs, dc2_train_augs_full
from deepdisc.data_format.image_readers import wlDC2ImageReader, wlDC2psfImageReader
from deepdisc.model.shear_models import CNNShearROIHeads, RedshiftPDFCasROIHeads, CNNShapeG_ROIHeads, CNNShape_psfROIHeads, CNNShapeG_weighted_ROIHeads, CNNShape_weighted_ROIHeads, CNNShapeG_weighted_cali_ROIHeads
from deepdisc.model.meta_arch import GeneralizedRCNNWCS

# Overrides
dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
#model.proposal_generator.batch_size_per_image=1024


model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512
model.roi_heads.positive_fraction = 0.5
model.backbone.bottom_up.in_chans = 6

model.pixel_mean = [
        0.05381286,
        0.04986344,
        0.07526361,
        0.10420945,
        0.14229655,
        0.21245764,
]
model.pixel_std = [
        2.9318833,
        1.8443471,
        2.581817,
        3.5950038,
        4.5809164,
        7.302009,
]

#model._target_ = meta_arch.GeneralizedRCNNWCS
model.backbone.square_pad = 525
model.roi_heads._target_ = CNNShearROIHeads #roiheads.RedshiftPDFCasROIHeads
model.roi_heads.shear_factor = 100**2
model.proposal_generator.pre_nms_topk=[10000,10000]
model.proposal_generator.post_nms_topk=[6000,3000]
#model.proposal_generator.pre_nms_topk=[200,1000]
#model.proposal_generator.post_nms_topk=[6000,6000]
#model.proposal_generator.nms_thresh = 0.3
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 3000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.5
    

train.init_checkpoint = "/projects/bdsp/wenyinli/models/shear_MSE/shear_MSE_50.pth"
#train.init_checkpoint = '/home/g4merz/DC2/model_tests/zoobot/zoobot_GZ2_resnet50_d2.pkl'

optimizer.lr = 0.0001
dataloader.test.mapper = loaders.ShapeMapper
dataloader.train.mapper = loaders.ShapeMapper

reader = wlDC2ImageReader()
dataloader.imagereader = reader

# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yacs_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
SOLVER.IMS_PER_BATCH = bs

DATASETS.TRAIN = "astro_train"
DATASETS.TEST = "astro_val"

SOLVER.BASE_LR = 0.0001
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


SOLVER.STEPS = []  # do not decay learning rate for retraining
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
TEST.DETECTIONS_PER_IMAGE = 3000
