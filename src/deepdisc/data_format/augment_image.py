"""Utilities for augmenting image data."""

import detectron2.data.transforms as T
import imgaug.augmenters as iaa
import numpy as np

import deepdisc.astrodet.detectron as detectron_addons
import random
import copy

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
            T.RandomRotation([0, 180, -90, 90], sample_style="choice"),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            #detectron_addons.CustomAug(multiband_gaussblur,prob=0.5),
        ],
        k= -1,
        cropaug=None,
        #cropaug=T.RandomCrop("relative", (0.5, 0.5))
    )
    return augs

