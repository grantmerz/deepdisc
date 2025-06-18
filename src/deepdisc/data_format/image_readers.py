import abc
import os

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb


class ImageReader(abc.ABC):
    """Base class that will read images on the fly for the training/testing dataloaders

    To implement an image reader for a new class, the derived class needs to have an
    __init__() function that calls super().__init__(*args, **kwargs)
    and a custom version of _read_image().
    """

    def __init__(self, norm="raw", *args, **kwargs):
        """
        Parameters
        ----------
        norm : str (optional)
            A contrast scaling to apply before data augmentation, i.e. luptonizing or z-score scaling
            Default = raw
        **kwargs : key word args
            Key word args for the contrast scaling function
        """
        self.scaling = ImageReader.norm_dict[norm]
        self.scalekwargs = kwargs

    @abc.abstractmethod
    def _read_image(self, key):
        """Read the image. No-op implementation.

        Parameters
        ----------
        key : str or int
            The key indicating the image to read.

        Returns
        -------
        im : numpy array
            The image.
        """
        pass

    def __call__(self, key):
        """Read the image and apply scaling.

        Parameters
        ----------
        key : str or int
            The key indicating the image to read.

        Returns
        -------
        im : numpy array
            The image.
        """
        im = self._read_image(key)
        im_scale = self.scaling(im, **self.scalekwargs)
        return im_scale

    def raw(im):
        """Apply raw image scaling (no scaling done).

        Parameters
        ----------
        im : numpy array
            The image.

        Returns
        -------
        numpy array
            The image with pixels as float32.
        """
        return im.astype(np.float32)

    def lupton(im, bandlist=[2, 1, 0], stretch=0.5, Q=10, m=0):
        """Apply Lupton scaling to the image and return the scaled image.

        Parameters
        ----------
        im : np array
            The image being scaled
        bandlist : list[int]
            Which bands to use for lupton scaling (must be 3)
        stretch : float
            lupton stretch parameter
        Q : float
            lupton Q parameter
        m: float
            lupton minimum parameter

        Returns
        -------
        image : numpy array
            The 3-channel image after lupton scaling using astropy make_lupton_rgb
        """
        assert np.array(im.shape).argmin() == 2 and len(bandlist) == 3
        b1 = im[:, :, bandlist[0]]
        b2 = im[:, :, bandlist[1]]
        b3 = im[:, :, bandlist[2]]

        return make_lupton_rgb(b1, b2, b3, minimum=m, stretch=stretch, Q=Q)

    def zscore(im, A=1., m=0.0):
        """Apply z-score scaling to the image and return the scaled image.

        Parameters
        ----------
        im : np array
            The image being scaled
        A : float
            A multiplicative scaling factor applied to each band
        m : float
            A minimum pixel value. Defaults to 0.0

        Returns
        -------
        image : numpy array
            The image after z-score scaling (subtract mean and divide by std deviation)
        """
        I = np.mean(im, axis=-1)
        Imean = np.nanmean(I)
        Isigma = np.nanstd(I)

        image = np.zeros_like(im)
        for i in range(im.shape[-1]):
            image[:, :, i] = A * (im[:, :, i] - Imean - m) / Isigma

        return image

    # This dict is created to map an input string to a scaling function
    norm_dict = {"raw": raw, "lupton": lupton, "zscore": zscore}

    @classmethod
    def add_scaling(cls, name, func):
        """Add a custom contrast scaling function

        ex)
        def sqrt(image):
            image[:,:,0] = np.sqrt(image[:,:,0])
            image[:,:,1] = np.sqrt(image[:,:,1])
            image[:,:,2] = np.sqrt(image[:,:,2])
            return image

        ImageReader.add_scaling('sqrt',sqrt)
        """
        cls.norm_dict[name] = func

