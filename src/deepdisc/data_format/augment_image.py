"""Utilities for augmenting image data."""

import detectron2.data.transforms as T
import numpy as np
import deepdisc.astrodet.detectron as detectron_addons
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

    if rng_seed is None:
        rng_seed = np.random.default_rng()

    new_ebv = rng_seed.uniform(0, 0.1)
    image = np.float32(image*(10.**(-A_EBV*new_ebv/2.5)))
    return image


def gaussblur(image,rng_seed=None):
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
 
    if rng_seed is None:
        rng_seed = np.random.default_rng()

    sigma = rng_seed.random()

    for i in range(image.shape[-1]):
        image[:, :, i] = gaussian_filter(
                        image[:, :, i], sigma, mode="mirror")

    return image


def scale_psf(sigi, lambda_eff):
    i_eff = 7546
    sig_lambda = sigi* (lambda_eff)**(-0.2)/(i_eff**(-0.2))
    return sig_lambda


def multiband_gaussblur(image,rng_seed=None):
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

    if rng_seed is None:
        rng_seed = np.random.default_rng()

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    sigmai = rng_seed.random()
    for i in range(image.shape[-1]):
        sigma = scale_psf(sigmai,LAMBDA_EFFS[i])
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
