"""Utilities for augmenting image data."""

import detectron2.data.transforms as T
import torchvision.transforms as torchvision_T
import imgaug.augmenters as iaa
import numpy as np

import deepdisc.astrodet.detectron as detectron_addons
import random

from PIL import Image


LAMBDA_EFFS = [3671,4827,6223,7546,8691,9712]
A_EBV = np.array([4.81,3.64,2.70,2.06,1.58,1.31])
# https://roman.gsfc.nasa.gov/science/WFI_technical.html#0.105b
# f184, h158, y106, j129
ROMAN_LAMBDA_EFFS = [18420,15770,10600,12930]
# ROMAN_LAMBDA_EFFS = random normal distribution of std=5*pixel scale using  the f184 band
# gal extinction if using redden aug


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
    new_ebv = np.random.uniform(0, 0.5)
    image = np.float32(image*(10.**(-A_EBV*new_ebv/2.5)))
    return image


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
    aug = iaa.GaussianBlur(sigma=10, seed=rng_seed)
    return aug.augment_image(image)


def scale_psf(sigi, lambda_eff, i_eff=7546):
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
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    imgs = np.zeros(image.shape, dtype=np.float32)
    sigmai = random.random()
    for i in range(6):
        sigma = scale_psf(sigmai,LAMBDA_EFFS[i])
        aug = iaa.GaussianBlur(sigma=sigma, seed=rng_seed)
        imaug = aug.augment_image(image[:,:,i])
        imgs[:,:,i] = imaug
    return imgs

def roman_gaussblur(image, rng_seed=None):
    """
    Temp function instead of changing multiband_gaussblur to handle n bands 
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
    imgs = np.zeros(image.shape, dtype=np.float32)
    sigmai = random.random()
    for i in range(4): # f184, h158, y106, j129
        sigma = scale_psf(sigmai,ROMAN_LAMBDA_EFFS[i], i_eff=10600) # adjusting the blur for each filter band based on its wavelength
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

def elastic_transform(image, alpha=100, sigma=10.0, rng_seed=None):
    if rng_seed is None:
        rng_seed = np.random.default_rng()
    imgs = np.zeros(image.shape, dtype=np.float32)
    for i in range(4):
        aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, seed=rng_seed)
        imaug = aug.augment_image(image[:,:,i])
        imgs[:,:,i] = imaug
    return imgs

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

    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            #detectron_addons.CustomAug(multiband_gaussblur,prob=1.0),
            #detectron_addons.CustomAug(redden,prob=1.0),

        ],
        k=-1,
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
            detectron_addons.CustomAug(multiband_gaussblur,prob=1.0),
            detectron_addons.CustomAug(redden,prob=1.0),

        ],
        k=-1,
        cropaug=None,
        #cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs


def hsc_test_augs(image):
    """Get the augmentation list

    Parameters
    ----------
    image: image
        The image to be augmented

    Returns
    -------
    augs: detectron2 AugmentationList
        The augs for hsc testing.  Set to 50% Crop due to memory constraints
    """
    augs = T.AugmentationList(
        [
            T.CropTransform(
                image.shape[1] // 4,
                image.shape[0] // 4,
                image.shape[1] // 2,
                image.shape[0] // 2,
            )
        ]
    )
    return augs

def roman_train_augs(image):
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            detectron_addons.CustomAug(roman_gaussblur,prob=1.0),
            detectron_addons.CustomAug(elastic_transform, prob=1.0),
            T.RandomBrightness(0.3,1.3),
            T.RandomContrast(0.8,1.2),
        ],
        k=-1,
         cropaug=None,
#        cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs

def roman_train_augs_gauss(image):
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            detectron_addons.CustomAug(roman_gaussblur,prob=1.0)
        ],
        k=-1,
         cropaug=None,
#        cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs

def roman_train_augs_elastic(image):
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            detectron_addons.CustomAug(elastic_transform, prob=1.0)
        ],
        k=-1,
         cropaug=None,
#        cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs

def roman_train_augs_ge(image):
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            detectron_addons.CustomAug(roman_gaussblur,prob=1.0),
            detectron_addons.CustomAug(elastic_transform, prob=1.0)
        ],
        k=-1,
         cropaug=None,
#        cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs


def roman_train_augs_cb(image):
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.3,1.3),
            T.RandomContrast(0.8,1.2),
        ],
        k=-1,
         cropaug=None,
#        cropaug=T.RandomCrop("relative", (0.5, 0.5))
        
    )
    return augs

def roman_train_focal(image):
    augs = detectron_addons.KRandomAugmentationList(
        [
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            detectron_addons.CustomAug(roman_gaussblur,prob=1.0),
        ],
        k=-1,
         cropaug=None,
#        cropaug=T.RandomCrop("relative", (0.5, 0.5))
        
    )
    return augs
