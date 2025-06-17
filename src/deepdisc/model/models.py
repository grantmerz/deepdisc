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
#from .fastrcnn import fast_rcnn_inference_noclip
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

        
class RedshiftPDFCasROIHeadsJWST(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        zmin: int,
        zmax: int,
        zn: int,
        output_features: bool,
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

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        self.zloss_factor = zloss_factor
        self.zmin = zmin
        self.zmax = zmax
        self.zn = zn
        self.output_features = output_features

        
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size)+1, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )
        

        self.sfd = SFDQuery()

        
        
    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances, targets=None, image_wcs=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            instances = []
            for x in finstances:
                z_inst = x[x.gt_redshift != -1]
                instances.append(z_inst)
            if len(instances)==0:
                return 0
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
        num_instances_per_img = [len(i) for i in instances]
        inds = np.cumsum(num_instances_per_img)

        
        #Add EBV
        centers = cat([box.get_centers().cpu() for box in boxes]) # Center box coords for wcs             
        #calculates coords for box centers in each image. Need to split and cumsum to make sure the box centers get the right wcs  
        coords = [WCS(wcs).pixel_to_world(np.split(centers,inds)[i][:,0],np.split(centers,inds)[i][:,1]) for i, wcs in enumerate(image_wcs)] 
        #use dustamps to get all ebv with the associated coords
        ebvvec = [torch.tensor(self.sfd(coordsi)).to(features.device) for coordsi in coords]
        ebvs = cat(ebvvec)
        #gather into a tensor and add as a feature for the input to the fully connected network
        features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)

        if self.training:
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor

            return {"redshift_loss": torch.mean(nlls)}

        else:
            if len(instances[0]) == 0 and len(instances)==1:
                for i, pred_instances in enumerate(instances):
                    pred_instances.pred_redshift_pdf=torch.tensor([]).to(features.device)
                    pred_instances.pred_gmm=torch.tensor([]).to(features.device)
                return instances
                
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(self.zmin, self.zmax, self.zn)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))

            inds = np.cumsum(num_instances_per_img)

            #probs = torch.zeros((torch.sum(nin), self.zn)).to(fcs.device)
            #for j, z in enumerate(zs):
            #    probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                #pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]
                if self.output_features:
                    pred_instances.features = np.split(features,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None, image_wcs=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets, image_wcs))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances, image_wcs=image_wcs)
            return pred_instances, {}


        
        


        
        
        
class RedshiftPDFCasROIHeadsGoldEBV(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Uses the image wcs and dustmaps to calculate ebv values for each box center 
        Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        maglim: float,
        output_features: bool,
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

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        self.zloss_factor = zloss_factor
        self.maglim = maglim
        self.output_features = output_features
        
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size)+1, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )
        
        
        self.sfd = SFDQuery()

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances, targets=None, image_wcs=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)

            instances = []
            for x in finstances:
                gold_inst = x[x.gt_magi < self.maglim]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
            
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
        num_instances_per_img = [len(i) for i in instances]
        inds = np.cumsum(num_instances_per_img)
        
        #Add EBV
        centers = cat([box.get_centers().cpu() for box in boxes]) # Center box coords for wcs             
        #calculates coords for box centers in each image. Need to split and cumsum to make sure the box centers get the right wcs  
        coords = [WCS(wcs).pixel_to_world(np.split(centers,inds)[i][:,0],np.split(centers,inds)[i][:,1]) for i, wcs in enumerate(image_wcs)] 
        #use dustamps to get all ebv with the associated coords
        ebvvec = [torch.tensor(self.sfd(coordsi)).to(features.device) for coordsi in coords]
        ebvs = cat(ebvvec)
        #gather into a tensor and add as a feature for the input to the fully connected network
        features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)

        if self.training:
            
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor

            return {"redshift_loss": torch.mean(nlls)}

        else:
            if len(instances[0]) == 0:
                return instances
            
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(0, 3, 300)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))


            probs = torch.zeros((torch.sum(nin), 300)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]
                if self.output_features:
                    pred_instances.features = np.split(features,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None, image_wcs=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets, image_wcs))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances, image_wcs=image_wcs)
            return pred_instances, {}

        
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

        
