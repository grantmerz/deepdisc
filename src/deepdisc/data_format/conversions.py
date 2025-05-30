import os

import astropy.io.fits as fits
import h5py
import numpy as np
import scarlet
from detectron2.utils.file_io import PathManager
from iopath.common.file_io import file_lock
import os, json, shutil
import logging
logger = logging.getLogger(__name__)

def fitsim_to_numpy(img_files, outdir):
    """Converts a list of single-band FITS images to multi-band numpy arrays

    Parameters
    ----------
    img_files: list[str]
        A nested list of the FITS image files.
        The first index is the image and the second index is the filter
    outdir: str
        The directory to output the numpy arrays


    """

    for images in img_files:
        full_im = []
        for img in images:
            with fits.open(img, memmap=False, lazy_load_hdus=False) as hdul:
                data = hdul[0].data
                full_im.append(data)

        full_im = np.array(full_im)
        
        
        np.save(os.path.join(outdir, img.split('_')[-3]+".npy"), full_im)


    return


def fitsim_to_hdf5(img_files, outname, dset="train"):
    """Converts a list of single-band FITS images to flattened multi-band images in an hdf5 file

    Parameters
    ----------
    img_files: list[str]
        A nested list of the FITS image files.
        The first index is the image and the second index is the filter
    outname: str
        The name of the output file
    

    """
    all_images = []
    for images in img_files:
        full_im = []
        for img in images:
            with fits.open(img, memmap=False, lazy_load_hdus=False) as hdul:
                data = hdul[0].data
                full_im.append(data)

        full_im = np.array(full_im)
        all_images.append(full_im.flatten())
    all_images = np.array(all_images)

    with h5py.File(outname, "w") as f:
        data = f.create_dataset("images", data=all_images)

    return


def ddict_to_hdf5(dataset_dicts, outname):
    """Converts a list of dataset dictionaries to an hdf5 file (for RAIL usage)

    Parameters
    ----------
    dataset_dicts: list[dict]
        The dataset dicts
    outname: str
        The name of the output file
    """
    
    with h5py.File(outname, 'w') as file:
        data = [json.dumps(this_dict) for this_dict in dataset_dicts]
        dt = h5py.special_dtype(vlen=str)
        dataset = file.create_dataset('metadata_dicts', data=data, dtype=dt)
        
    return


def numpyim_to_hdf5(img_files, outname):
    """Converts a list of single-band FITS images to flattened multi-band images in an hdf5 file

    Parameters
    ----------
    img_files: list[str]
        A nested list of the FITS image files.
        The first index is the image and the second index is the filter
    outname: str
        The name of the output file
    

    """
    all_images = []
    for img_file in img_files:
        full_im = np.load(img_file)
        all_images.append(full_im.flatten())
    all_images = np.array(all_images)

    with h5py.File(outname, "w") as f:
        data = f.create_dataset("images", data=all_images)

    return


def convert_to_json(dict_list, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            print(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(dict_list, f)
            shutil.move(tmp_file, output_file)
