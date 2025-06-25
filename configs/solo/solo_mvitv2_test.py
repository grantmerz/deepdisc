""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from omegaconf import OmegaConf
import numpy as np
# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs = 1

metadata = OmegaConf.create() 
metadata.classes = ["object"]

numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..COCO.cascade_mask_rcnn_mvitv2_b_in21k_100ep import dataloader, model, train, lr_multiplier, optimizer
import deepdisc.model.models as roiheads
import deepdisc.model.loaders as loaders
import deepdisc.model.meta_arch as meta_arch 
from deepdisc.data_format.augment_image import dc2_train_augs, dc2_train_augs_full
from deepdisc.data_format.image_readers import DC2ImageReader

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

#for 0.9 resample
#model.pixel_mean = [0.00249514, 0.00338973, 0.00421167, 0.00498388, 0.00514987, 0.00259173, 0.00467681, 0.00446813, 0.00451128]
#model.pixel_std = [0.0467363, 0.06604525, 0.07411944, 0.08065032, 0.06442349, 0.03459511, 0.05869897, 0.06095179, 0.06141144]

#model.backbone.square_pad = 512

model.roi_heads.num_components = 5
model.roi_heads.zloss_factor = 1
#model.roi_heads.zbins = np.linspace(0,3,300)
#model.roi_heads.weights = np.load('/home/g4merz/rail_deepdisc/configs/solo/zweights.npy')
#model.roi_heads.maglim = 25.3
model.roi_heads.zmin = 0
model.roi_heads.zmax = 11
model.roi_heads.zn = 1100

#model._target_ = meta_arch.GeneralizedRCNNWCS
model.roi_heads._target_ = roiheads.RedshiftPDFCasROIHeads

model.proposal_generator.pre_nms_topk=[10000,10000]
model.proposal_generator.post_nms_topk=[6000,3000]
#model.proposal_generator.pre_nms_topk=[200,1000]
#model.proposal_generator.post_nms_topk=[6000,6000]
#model.proposal_generator.nms_thresh = 0.3
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 3000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.5
    

train.init_checkpoint = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_8c3da3.pkl"
#train.init_checkpoint = "/home/g4merz/JWST/models/MViTv2_NG5_newBB_15.pth"

optimizer.lr = 0.001
#dataloader.test.mapper = loaders.RedshiftDictmapper
dataloader.train.mapper = loaders.RedshiftDictMapper

reader = DC2ImageReader()
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
