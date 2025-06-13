from typing import Dict, List, Optional, Tuple
import os
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
#from .fastrcnn import fast_rcnn_inference_noclip
from torch import nn
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.nn import functional as F
from deepdisc.preprocessing.get_data import get_psf_itpl

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

    model.to(cfg.train.device)
    model = create_ddp_model(model, **cfg.train.ddp)

    return model

class HSC_CNNShapeG_weighted_ROIHeads(CascadeROIHeads):

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        shape_factor,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.shape_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.shape_factor = shape_factor
        self.shape_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 1024, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024*3*3, 1024),
            nn.ReLU(),
            #nn.Dropout(p=0.5),  
            nn.Linear(1024, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.5), 
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def _forward_shape(self, features, instances, targets=None):
        if self.training:
            proposals = add_ground_truth_to_proposals(targets, instances)
            instances, _ = select_foreground_proposals(proposals, self.num_classes)
        
        if self.shape_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.shape_pooler(features, boxes)
        
        #features = nn.Flatten()(features)
        #ebvs = cat([x.gt_ebv for x in instances])
        #features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)

        
        num_instances_per_img = [len(i) for i in instances]
        
        instances

        if self.training:
            outputs = self.shape_conv(features)
            shapes = outputs[:,:2]
            log_vari = outputs[:,2:]
            vari = torch.exp(log_vari)
            gt_shapes = torch.stack((cat([x.gt_et_1 for x in instances]),
                                     cat([x.gt_et_2 for x in instances]),
                                     ),dim = 1)
            has_shape = torch.tensor(cat([x.gt_has_shape for x in instances])).bool()
            star_mask = torch.tensor(cat([x.gt_c_id for x in instances])).bool()

            chi = (gt_shapes-shapes)**2/vari*0.5 + 0.5*log_vari # [n, 2]
            weight = torch.tensor(cat([x.gt_e_weight for x in instances])).unsqueeze(1)
            chi = chi[has_shape]* weight[has_shape]
            loss = torch.mean(chi)*self.shape_factor
            star_loss = torch.mean(shapes[star_mask]**2)

            return {"shape_WMSEloss": loss, "shape_star_loss": star_loss}

        else:
            if len(instances[0]) == 0:
                return instances
            #print(features.mean(dim = [2,3]))
            shapes = self.shape_conv(features)
            for i, pred_instances in enumerate(instances):
                pred_instances.shapes = shapes
                
            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            
        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_shape(features, proposals, targets))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_shape(features, pred_instances)
            return pred_instances, {}