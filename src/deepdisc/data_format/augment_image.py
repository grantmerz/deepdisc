"""Utilities for augmenting image data."""

import detectron2.data.transforms as T
import imgaug.augmenters as iaa
import numpy as np

import deepdisc.astrodet.detectron as detectron_addons
import random
import copy
    
'''def trans_shape(instances, transforms):
    for t in transforms:
        try:
            instances = t.apply_rotation(instances)
        except:
            continue
    return instances'''

def flip_e(self, instances):
    et_1 = instances.get("gt_et_1")
    et_2 = instances.get("gt_et_2")
    #shear_1 = instances.get("gt_shear_1")
    #shear_2 = instances.get("gt_shear_2")
    instances.set("gt_et_2", -et_2)
    #instances.set("gt_shear_2", -shear_2)
    return instances

def rotate_e(self, instances):
    angle = self.angle/180*np.pi
    et_1 = instances.get("gt_et_1")
    et_2 = instances.get("gt_et_2")
    et_1_r = et_1*np.cos(angle*2)-et_2*np.sin(angle*2)
    et_2_r = et_1*np.sin(angle*2)+et_2*np.cos(angle*2)
    
    #shear_1 = instances.get("gt_shear_1")
    #shear_2 = instances.get("gt_shear_2")
    #shear_1_r = shear_1*np.cos(angle*2)-shear_2*np.sin(angle*2)
    #shear_2_r = shear_1*np.sin(angle*2)+shear_2*np.cos(angle*2)

    instances.set("gt_et_1", et_1_r)
    instances.set("gt_et_2", et_2_r)
    #instances.set("gt_shear_1", shear_1_r)
    #instances.set("gt_shear_2", shear_2_r)
    return instances

T.VFlipTransform.apply_rotation = flip_e
T.HFlipTransform.apply_rotation = flip_e
T.RotationTransform.apply_rotation = rotate_e

LAMBDA_EFFS = [3671,4827,6223,7546,8691,9712]
A_EBV = np.array([4.81,3.64,2.70,2.06,1.58,1.31])



def redden(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    new_ebv = np.random.uniform(0, 0.1)
    image = np.float32(image*(10.**(-A_EBV*new_ebv/2.5)))
    return image



def filter_dropout(image):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    image_drop = copy.copy(image)
    filt = np.random.choice(np.arange(0,image.shape[-1]))
    image_drop[:,:,filt] = np.zeros(image.shape[:-1])
    if np.all(image_drop==0):
        return image
    else:
        return image_drop

def gaussblur(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.GaussianBlur(sigma=50, seed=rng_seed)
    return aug.augment_image(image)


def scale_psf(sigi, lambda_eff):
    i_eff = 7546
    sig_lambda = sigi* (lambda_eff)**(-0.3)/(i_eff**(-0.3))
    return sig_lambda


def multiband_gaussblur(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    imgs = np.zeros(image.shape, dtype=np.float32)
    sigmai = random.random()
    for i in range(image.shape[-1]):
        sigma = scale_psf(sigmai,LAMBDA_EFFS[3])
        aug = iaa.GaussianBlur(sigma=sigma, seed=rng_seed)
        imaug = aug.augment_image(image[:,:,i])
        imgs[:,:,i] = imaug
    return imgs

def addelementwise16(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.AddElementwise((-3276, 3276), seed=rng_seed)
    return aug.augment_image(image)


def addelementwise8(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.AddElementwise((-25, 25), seed=rng_seed)
    return aug.augment_image(image)


def addelementwise(image, rng_seed=None):
    """
    Parameters
    ----------
    image: ndarray
    rng_seed : np.random.Generator
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    augmented image

    """
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    aug = iaa.AddElementwise((-image.max() * 0.1, image.max() * 0.1), seed=rng_seed)
    return aug.augment_image(image)


def centercrop(image):
    """Crop an image to just the center portion

    Parameters
    ----------
    image: ndarray

    Returns
    -------
    cropped image
    """
    h, w = image.shape[:2]
    hc = (h - h // 2) // 2
    wc = (w - w // 2) // 2
    image = image[hc : hc + h // 2, wc : wc + w // 2]
    return image


def train_augs(image):
    """Get the augmentation list

    Parameters
    ----------
    image: image
        The image to be augmented

    Returns
    -------
    augs: detectron_addons.KRandomAugmentationList
        The list of augs for training.  Set to RandomRotation, RandomFlip, RandomCrop
    """

    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ],
        k=-1,
        cropaug=T.RandomCrop("relative", (0.5, 0.5)),
    )
    return augs


def dc2_train_augs(image):
    """Get the augmentation list

    Parameters
    ----------
    image: image
        The image to be augmented

    Returns
    -------
    augs: detectron_addons.KRandomAugmentationList
        The list of augs for training.  Set to RandomRotation, RandomFlip, RandomCrop
    """
    #T.VFlipTransform.apply_rotation = flip_e
    #T.HFlipTransform.apply_rotation = flip_e
    #T.RotationTransform.apply_rotation = rotate_e

    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([45, -45, 0], sample_style="choice"),
            T.RandomRotation([0, 180, -90, 90], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            #detectron_addons.CustomAug(multiband_gaussblur,prob=0.5),
        ],
        k= None,
        cropaug=None,
        #cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs

def dc2_val_augs(image):
    """Get the augmentation list

    Parameters
    ----------
    image: image
        The image to be augmented

    Returns
    -------
    augs: detectron_addons.KRandomAugmentationList
        The list of augs for training.  Set to RandomRotation, RandomFlip, RandomCrop
    """

    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
        ],
        k= None,
        cropaug=None,
        #cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs



def dc2_train_augs_full(image):
    """Get the augmentation list

    Parameters
    ----------
    image: image
        The image to be augmented

    Returns
    -------
    augs: detectron_addons.KRandomAugmentationList
        The list of augs for training.  Set to RandomRotation, RandomFlip, RandomCrop
    """

    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            #detectron_addons.CustomAug(multiband_gaussblur,prob=1.0),
            detectron_addons.CustomAug(redden,prob=1.0),

        ],
        k=-1,
        cropaug=None,
        #cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs


def jwst_dropout_augs(image):
    """Get the augmentation list

    Parameters
    ----------
    image: image
        The image to be augmented

    Returns
    -------
    augs: detectron_addons.KRandomAugmentationList
        The list of augs for training.  Set to RandomRotation, RandomFlip, RandomCrop
    """

    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            detectron_addons.CustomAug(filter_dropout,prob=1.0),
            #detectron_addons.CustomAug(redden,prob=1.0),

        ],
        k=-1,
        cropaug=None,
        #cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs
