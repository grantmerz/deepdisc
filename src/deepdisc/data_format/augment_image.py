"""Utilities for augmenting image data."""

import detectron2.data.transforms as T
import numpy as np

import deepdisc.astrodet.detectron as detectron_addons
import random
import copy
from scipy.ndimage import gaussian_filter


LAMBDA_EFFS = [3671,4827,6223,7546,8691,9712]
A_EBV = np.array([4.81,3.64,2.70,2.06,1.58,1.31])



def redden(image, rng_seed=None):
    """

    Redden the image based off of A_E(B-V) values

    Parameters
    ----------
    image: ndarray HxWxC
    
    Returns
    -------
    augmented image

    """
    new_ebv = np.random.uniform(0, 0.1)
    image = np.float32(image*(10.**(-A_EBV*new_ebv/2.5)))
    return image


def filter_dropout(image):
    """

    Randomly drop out a filter

    Parameters
    ----------
    image: ndarray HxWxC
    
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

def gaussblur(image):
    """

    Convolve with a gaussian filter to mimic psf blurring
    Assumes constant (achromatic) psf 

    Parameters
    ----------
    image: ndarray
    
    Returns
    -------
    augmented image

    """
 
    sigma = np.random.random()

    for i in range(image.shape[-1]):
        image[:, :, i] = gaussian_filter(
                        image[:, :, i], sigma, mode="mirror")

    return image


def scale_psf(sigi, lambda_eff):
    i_eff = 7546
    sig_lambda = sigi* (lambda_eff)**(-0.3)/(i_eff**(-0.3))
    return sig_lambda


def multiband_gaussblur(image):
    """

    Convolve with a gaussian filter to mimic psf blurring
    PSF size changes across filters based on the scale_psf function 
    
    Parameters
    ----------
    image: ndarray
   
    Returns
    -------
    augmented image

    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    sigmai = random.random()
    for i in range(image.shape[-1]):
        sigma = scale_psf(sigmai,LAMBDA_EFFS[3])
        image[:, :, i] = gaussian_filter(
                image[:, :, i], sigma, mode="mirror")
    return image


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
