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

    def __call__(self, image):
        """Read the image and apply scaling.

        Parameters
        ----------
        image : str or numpy array
            The path indicating the image to read or image data in a numpy array with dimensions (band, h, w).

        Returns
        -------
        im : numpy array
            The image.
        """
        if isinstance(image, str) or all(isinstance(s, str) for s in image):
            im = self._read_image(image)
        elif isinstance(image, np.ndarray):
            im = np.transpose(image, axes=(1, 2, 0)).astype(np.float32)
        else:
            raise ValueError("Input must be a string or a numpy array.")
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

class wlDC2ImageReader(ImageReader):
    """An ImageReader for DC2 image files."""

    def __init__(self, bands = ['u', 'g', 'r', 'i', 'z', 'y'], *args, **kwargs):
        self.bands = bands
        #print(self.bands)
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def _read_image(self, filename):
        """Read the image.

        Parameters
        ----------
        filename : str
            The filename indicating the image to read.

        Returns
        -------
        im : numpy array
            The image.
        """
        #bands = ['g', 'r', 'i', 'z', 'y']
        eg = fits.getdata(os.path.join(filename + "_"+self.bands[0]+".fits"), memmap=False)
        length, width = eg.shape
        image = np.empty([length, width, len(self.bands)], dtype=np.float64)
        for (i, band) in enumerate(self.bands):
            image[:,:,i] = fits.getdata(os.path.join(filename + "_"+band+".fits"), memmap=False).astype('float64')
        #print(image.shape)
        return image.astype('float32')
    
class wlDC2psfImageReader(ImageReader):
    """An ImageReader for DC2 image files."""

    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def _read_image(self, filename):
        """Read the image.

        Parameters
        ----------
        filename : str
            The filename indicating the image to read.

        Returns
        -------
        im : numpy array
            The image.
        """
        filters = ['u','g','r','i','z','y']
        filter_psfs = [i+'_psfs' for i in filters]
        filters += filter_psfs
        g = fits.getdata(os.path.join(filename + "_g.fits"), memmap=False)
        length, width = g.shape
        image = np.empty([length, width, len(filters)], dtype=np.float64)
        for (i, band) in enumerate(filters):
            image[:,:,i] = fits.getdata(os.path.join(filename + "_"+band+".fits"), memmap=False).astype('float64')
        
        return image.astype('float32')

class wlHSCImageReader(ImageReader):
    """An ImageReader for DC2 image files."""

    def __init__(self, bands, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)
        self.bands = bands
        
    def _read_image(self, filename):
        """Read the image.

        Parameters
        ----------
        filename : str
            The filename indicating the image to read.

        Returns
        -------
        im : numpy array
            The image.
        """
        #filters = ['G', 'R', 'I', 'Z', 'Y']
        eg = fits.getdata(os.path.join(filename + "_"+self.bands[0]+".fits"), memmap=False)
        length, width = eg.shape
        image = np.empty([length, width, len(self.bands)], dtype=np.float64)
        for (i, band) in enumerate(self.bands):
            image[:,:,i] = fits.getdata(os.path.join(filename + "_"+band+".fits"), memmap=False).astype('float64')
        
        return image.astype('float32')


class NumpyImageReader(ImageReader):
    """An ImageReader for DC2 image files."""

    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def _read_image(self, filename):
        """Read the image.

        Parameters
        ----------
        filename : str
            The filename indicating the image to read.

        Returns
        -------
        im : numpy array
            The image.
        """
        file = filename.split("/")[-1].split(".")[0]
        base = os.path.dirname(filename)
        fn = os.path.join(base, file) + ".npy"
        image = np.load(fn)
        image = np.transpose(image, axes=(1, 2, 0)).astype(np.float32)
        return image
