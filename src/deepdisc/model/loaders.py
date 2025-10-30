import copy

import detectron2.data as data
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts

import deepdisc.astrodet.astrodet as toolkit
import deepdisc.astrodet.detectron as detectron_addons
from astropy.wcs import WCS
import h5py
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as torchdata
from detectron2.data.samplers import TrainingSampler
#from detectron2.data.common ToIterableDataset

def trans_shape(instances, transforms):
    for t in transforms:
        try:
            instances = t.apply_rotation(instances)
        except:
            continue
    return instances


class DataMapper:
    """Base class that will map data to the format necessary for the model

    To implement a data mapper for a new class, the derived class needs to have an
    __init__() function that calls super().__init__(*args, **kwargs)
    and a custom version of map_data().
    """

    def __init__(self, imreader=None, key_mapper=None, augmentations=None):
        """
        Parameters
        ----------
        imreader : ImageReader
            The class that will load and contrast scale the images.
            They can be stored separately from the dataset or with it.
        key_mapper : function
            The function that takes the data set and returns the key that will be used to load the image.
            If the image is stored with the dataset, this is not needed
            Default = None
        augmentations : detectron2 AugmentationList or a detectron_addons.KRandomAugmentationList
            The list of augmentations to apply to the image
            Default = None
        """
        self.IR = imreader
        self.km = key_mapper
        self.augmentations = augmentations

    def map_data(self, data):
        return data


class DictMapper(DataMapper):
    """Class that will map COCO dictionary data to the format necessary for the model"""

    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])
        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
        }

    

def return_train_loader(cfg, mapper):
    """Returns a train loader

    Parameters
    ----------
    cfg : LazyConfig
        The lazy config, which contains data loader config values

    **kwargs for the read_image functionality

    Returns
    -------
        a train loader
    """
    loader = data.build_detection_train_loader(cfg, mapper=mapper)
    return loader

def return_test_loader(cfg, mapper):
    """Returns a test loader with configurable batch size

    Parameters
    ----------
    cfg : LazyConfig
        The lazy config, which contains data loader config values
        including batch size and num_workers
    **kwargs for the read_image functionality

    Returns
    -------
        a test loader
    """
    batch_size = getattr(cfg.dataloader.test, 'total_batch_size', 1)
    num_workers = getattr(cfg.dataloader.test, 'num_workers', 0)
    dataset = get_detection_dataset_dicts(cfg.DATASETS.TEST)
    # loader w/ explicit params so we can bypass @configurable for detectron2's build_detection_test_loader()
    loader = data.build_detection_test_loader(
        dataset,
        mapper=mapper,
        batch_size=batch_size,   
        num_workers=num_workers, 
    )
    return loader

def return_custom_train_loader(dataset,batch_size=4, distributed=False):
    
    if distributed:
        datasetI = ToIterableDataset(dataset, sampler, shard_chunk_size=batch_size)
        
        loader = torchdata.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    drop_last=True,
                    num_workers=0,
                    #worker_init_fn=worker_init_reset_seed,
                    #prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    persistent_workers=False,
                    pin_memory=True,
                    sampler=torchdata.distributed.DistributedSampler(dataset,shuffle=True)
                )
    
    else:
        loader = torchdata.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )

    return loader

