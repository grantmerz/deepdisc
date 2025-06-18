from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm, nonzero_tuple
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference
from detectron2.modeling.roi_heads import CascadeROIHeads, StandardROIHeads, select_foreground_proposals
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from torch import nn
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.nn import functional as F

import dustmaps
from dustmaps.sfd import SFDQuery

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord



def return_lazy_model(cfg, freeze=True):
    """Return a model formed from a LazyConfig with the backbone
    frozen. Only the head layers will be trained.

    Parameters
    ----------
    cfg : .py file
        a LazyConfig

    Returns
    -------
        torch model
    """
    model = instantiate(cfg.model)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        # Phase 1: Unfreeze only the roi_heads
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        # Phase 2: Unfreeze region proposal generator with reduced lr
        for param in model.proposal_generator.parameters():
            param.requires_grad = True
        for param in model.backbone.bottom_up.stem.parameters():
            param.requires_grad= True

    model.to(cfg.train.device)
    model = create_ddp_model(model, **cfg.train.ddp)

    return model

        
class RedshiftProjectionHead(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.redshift_head = nn.Sequential(
                nn.Linear(np.prod(input_shape), 128),
                nn.PReLU(),
                nn.Linear(128, 1),
                #nn.ReLU()
            )
        
    def forward(self, features, labels=None):    
        features = nn.Flatten()(features)
        pred_z = self.redshift_head(features)

        if self.training:
            labels = labels.unsqueeze(1)
            #print(pred_z[:,0])
            diff = (pred_z - labels)
            #print(diff)
            return {"redshift_loss": torch.square(diff).mean()}

        else:
            return pred_z

        
