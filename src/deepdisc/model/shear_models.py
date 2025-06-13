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

class CNNShape_psfROIHeads(CascadeROIHeads):

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        shape_factor,
        n_blocks = 16,
        bands = ['u', 'g', 'r', 'i', 'z', 'y'],
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
        self.bands = bands
        self.n_blocks = n_blocks
        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.shape_factor = shape_factor
        self.shape_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 1024, stride=1,kernel_size=3),
            nn.ReLU(),
        )
        self.shape_mlp = nn.Sequential(
            nn.Linear(1024*3*3+18, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            
        )

        self.mag_mlp = nn.Sequential(
            nn.Linear(1024*3*3+18, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )


    def _forward_shape(self, features, instances, targets=None, filename = None):
        if self.training:
            proposals = add_ground_truth_to_proposals(targets, instances)
            instances, _ = select_foreground_proposals(proposals, self.num_classes)
        
        if self.shape_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.shape_pooler(features, boxes)
        
        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            psf = cat([x.gt_psfs for x in instances], dim = 1)
            conv_outputs = self.shape_conv(features)
            mlp_input = nn.Flatten()(conv_outputs)
            mlp_input = torch.cat((mlp_input, psf), dim = 1)
            outputs = self.shape_mlp(mlp_input)
            
            shapes = outputs[:,:2]
            log_vari = outputs[:,2:]
            vari = torch.exp(log_vari)
            gt_shapes = torch.stack((cat([x.gt_et_1 for x in instances]),
                                     cat([x.gt_et_2 for x in instances]),
                                     ),dim = 1)

            chi = (gt_shapes-shapes)**2/vari*0.5 + 0.5*log_vari 
            loss = torch.mean(chi)*self.shape_factor

            mag_ouputs = self.mag_mlp(mlp_input)
            gt_mag = torch.tensor(cat([x.gt_mag_i for x in instances]))
            mag_loss = torch.mean((gt_mag-mag_ouputs)**2)
                                

            return {"shape_chi_loss": loss, "mag_MSEloss": mag_loss}

        else:
            if len(instances[0]) == 0:
                return instances
            tract = filename[0].split(os.sep)[-4]
            patch = filename[0].split(os.sep)[-3]
            sp = filename[0].split(os.sep)[-2]
            centers = torch.tensor(np.array(cat([box.get_centers().cpu() for box in boxes])))
            psf = torch.zeros(len(features), 18).to(features.device).to(torch.float32)
            for (i,band) in enumerate(self.bands):
                cutout = get_psf_itpl(tract, patch, self.n_blocks, int(sp), band)
                psf_temp = torch.tensor(np.array([cutout[:,pos[1].int(), pos[0].int()] for pos in centers])).to(features.device).to(torch.float32)
                psf[:,(3*i):(3*(i+1))] = psf_temp
            
            conv_outputs = self.shape_conv(features)
            mlp_input = nn.Flatten()(conv_outputs)
            mlp_input = torch.cat((mlp_input, psf), dim = 1)
            outputs = self.shape_mlp(mlp_input)
            shapes = outputs
            
            
            
            for i, pred_instances in enumerate(instances):
                pred_instances.shapes = shapes
            return instances

    def forward(self, images, features, proposals, targets=None, filename = None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            
        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_shape(features, proposals, targets, filename))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_shape(features, pred_instances, targets, filename)
            return pred_instances, {}

class CNNShape_ROIHeads(CascadeROIHeads):

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
            nn.Conv2d(512, 1024, stride=1,kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            
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

        if self.training:
            outputs = self.shape_conv(features)
            shapes = outputs[:,:2]
            gt_shapes = torch.stack((cat([x.gt_et_1 for x in instances]),
                                     cat([x.gt_et_2 for x in instances]),
                                     ),dim = 1)
            chi = (gt_shapes-shapes)**2
            chi = chi
            loss = torch.mean(chi)*self.shape_factor

            return {"shape_MSEloss": loss}

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
        
class CNNShapeG_weighted_ROIHeads(CascadeROIHeads):

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
        inds = np.cumsum(num_instances_per_img)
        if self.training:
            outputs = self.shape_conv(features)

            shapes = outputs[:,:2]
            log_vari = outputs[:,2:]
            vari = torch.exp(log_vari)

            gt_shapes = torch.stack((cat([x.gt_et_1 for x in instances]),
                                     cat([x.gt_et_2 for x in instances]),
                                     ),dim = 1)
            #gt_size_1 = torch.tensor(cat([x.gt_size_1 for x in instances]))
            gt_mag_i = torch.tensor(cat([x.gt_mag_i for x in instances]))
            c_id = (1-torch.tensor(cat([x.gt_c_id for x in instances]))).bool()
            star_mask = torch.tensor(cat([x.gt_c_id for x in instances])).bool()
            chi = (gt_shapes-shapes)**2/vari*0.5 + 0.5*log_vari
            #weight = (nn.Sigmoid()((gt_size_1-0.2)*2)*nn.Sigmoid()(25-gt_mag_i)).unsqueeze(1)
            weight = (nn.Sigmoid()(25-gt_mag_i)).unsqueeze(1)
            chi = chi*weight

            loss = torch.mean(chi[c_id])*self.shape_factor
            star_loss = torch.mean((shapes[star_mask]-gt_shapes[star_mask])**2)

            if torch.isnan(star_loss):
                star_loss = torch.tensor(0.0).to(features.device)

            return {"shape_WMSEloss": loss}

        else:
            if len(instances[0]) == 0:
                return instances
            #print(features.mean(dim = [2,3]))
            shapes = self.shape_conv(features)
            #print(shapes.shape)
            shapes = np.split(shapes,inds)
            for i, pred_instances in enumerate(instances):
                pred_instances.shapes = shapes[i]
                
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
        
class CNNShapeG_weighted_cali_ROIHeads(CascadeROIHeads):

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
        
        self.cali_conv = nn.Sequential(
                        nn.Linear(4, 16),
                        nn.ReLU(),
                        nn.Linear(16, 4),
                        nn.ReLU(),
                        nn.Linear(4, 2),
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

        if self.training:
            outputs = self.shape_conv(features) # [n, 4]
            outputs_cali = self.cali_conv(outputs)
            
            shapes = outputs[:,:2] 
            log_vari = outputs[:,2:] 
            vari = torch.exp(log_vari)
            gt_shapes = torch.stack((cat([x.gt_et_1 for x in instances]), #[n, 2]
                                     cat([x.gt_et_2 for x in instances]),
                                     ),dim = 1) 
            gt_abs_e = torch.sqrt(gt_shapes[:,0]**2+gt_shapes[:,1]**2)
            
            gt_size_1 = torch.tensor(cat([x.gt_size_1 for x in instances]))
            gt_mag_i = torch.tensor(cat([x.gt_mag_i for x in instances]))
            chi = (gt_shapes-shapes)**2/vari*0.5 + 0.5*log_vari
            weight = (nn.Sigmoid()((gt_size_1-0.3)*4)*nn.Sigmoid()(5*gt_abs_e-4)).unsqueeze(1)
            weight[gt_size_1<=0.2] = 0
            weight[gt_mag_i>=25.3] = 0
            
            chi = chi* weight
            loss = torch.mean(chi)*self.shape_factor
            
            weight_cali = nn.ReLU()(25.3-gt_mag_i).unsqueeze(1)
            loss_cali = torch.mean((outputs_cali-gt_shapes)**2*weight_cali)*self.shape_factor

            return {"shape_WMSEloss": loss, "shape_CALIloss": loss_cali}

        else:
            if len(instances[0]) == 0:
                return instances
            #print(features.mean(dim = [2,3]))
            shapes = self.shape_conv(features)
            shapes_cali = self.cali_conv(shapes)
            for i, pred_instances in enumerate(instances):
                pred_instances.shapes = shapes
                pred_instances.shapes_cali = shapes_cali
                
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

class CNNShapeG_weighted_mag_ROIHeads(CascadeROIHeads):

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

        self.mag_conv = nn.Sequential(
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
            nn.Linear(64, 1),
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

        if self.training:
            outputs = self.shape_conv(features)
            mag_outputs = self.mag_conv(features)

            shapes = outputs[:,:2]
            log_vari = outputs[:,2:]
            vari = torch.exp(log_vari)

            mag_i = mag_outputs[:,:1]

            gt_shapes = torch.stack((cat([x.gt_et_1 for x in instances]),
                                     cat([x.gt_et_2 for x in instances]),
                                     ),dim = 1)
            #gt_size_1 = torch.tensor(cat([x.gt_size_1 for x in instances]))
            gt_mag_i = torch.tensor(cat([x.gt_mag_i for x in instances]))
            c_id = (1-torch.tensor(cat([x.gt_c_id for x in instances]))).bool()
            star_mask = torch.tensor(cat([x.gt_c_id for x in instances])).bool()
            chi = (gt_shapes-shapes)**2/vari*0.5 + 0.5*log_vari
            #weight = (nn.Sigmoid()((gt_size_1-0.2)*2)*nn.Sigmoid()(25-gt_mag_i)).unsqueeze(1)
            weight = (nn.Sigmoid()(25-gt_mag_i)).unsqueeze(1)
            chi = chi* weight
            
            loss = torch.mean(chi[c_id])*self.shape_factor
            mag_loss = torch.mean((mag_i[c_id]-gt_mag_i[c_id])**2)
            star_loss = torch.mean((shapes[star_mask]-gt_shapes[star_mask])**2)


            return {"shape_WMSEloss": loss, "shape_star_loss": star_loss, "mag_loss": mag_loss}

        else:
            if len(instances[0]) == 0:
                return instances
            #print(features.mean(dim = [2,3]))
            shapes = self.shape_conv(features)
            mag_i = self.mag_conv(features)
            for i, pred_instances in enumerate(instances):
                pred_instances.shapes = shapes
                pred_instances.mag_i = mag_i
                
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

class CNNShapeG_ROIHeads(CascadeROIHeads):

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
            nn.Conv2d(512, 1024, stride=1,kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
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

        if self.training:
            outputs = self.shape_conv(features)
            shapes = outputs[:,:2]
            log_vari = outputs[:,2:]
            vari = torch.exp(log_vari)
            gt_shapes = torch.stack((cat([x.gt_et_1 for x in instances]),
                                     cat([x.gt_et_2 for x in instances]),
                                     ),dim = 1)
            chi = (gt_shapes-shapes)**2/vari*0.5 + 0.5*log_vari 
            loss = torch.mean(chi)*self.shape_factor
            return {"shape_MSEloss": loss}

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
        
        
class CNNShearROIHeads(CascadeROIHeads):

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        shear_factor,
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

        self.shear_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.shear_factor = shear_factor
        print(self.shear_factor)
        self.shear_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 1024, stride=1,kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),

        )

    def _forward_shear(self, features, instances, targets=None):
        if self.training:
            proposals = add_ground_truth_to_proposals(targets, instances)
            instances, _ = select_foreground_proposals(proposals, self.num_classes)
        
        if self.shear_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.shear_pooler(features, boxes)
        
        #features = nn.Flatten()(features)
        #ebvs = cat([x.gt_ebv for x in instances])
        #features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)

        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            shears = self.shear_conv(features)
            #print(shears.shape)
            #with torch.no_grad():
            #    print(torch.std_mean(shears, dim = 0))
            gt_shears = torch.stack((cat([x.gt_shear_1 for x in instances]),
                                     cat([x.gt_shear_2 for x in instances]),
                                     ),dim = 1)
            gt_mag_i = torch.tensor(cat([x.gt_mag_i for x in instances]))
            weight = (nn.ReLU()(23-gt_mag_i)).unsqueeze(1)
            #gt_shears = cat([x.gt_redshift for x in instances])
            #print(gt_shears.shape)
            #print(((gt_shears-shears)**2).shape)
            #print(torch.std_mean(gt_shears, dim = 0))
            loss = torch.mean((gt_shears-shears)**2*weight)/weight.mean()*self.shear_factor
            #print(weight.mean())
            return {"shear_MSEloss": loss}

        else:
            if len(instances[0]) == 0:
                return instances
            #print(features.mean(dim = [2,3]))
            shears = self.shear_conv(features)
            #print('shear:')
            #print(shears)
            for i, pred_instances in enumerate(instances):
                pred_instances.shears = shears
                
            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            
        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_shear(features, proposals, targets))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_shear(features, pred_instances)
            return pred_instances, {}
