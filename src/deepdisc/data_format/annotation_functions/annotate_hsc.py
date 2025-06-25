import cv2
import numpy as np
from astropy.io import fits
from detectron2.structures import BoxMode
import os

# This is primarily a reference, no need to change.
FILT_INX = 0  # g=0, r=1, i=2


def annotate_hsc(images, mask, idx, filters):
    """Generates annotation metadata for hsc data

    Parameters
    ----------
    images : list
        A list of paths to image files, expected to have one file per filter.
    mask: str
        A path to a mask file for the images.
    idx: int
        An integer to uniquely identify the resulting record.
    filters: list
        A list of all filter labels, should map to the list of images.

    Returns
    -------
    record : dictionary
        A dictionary of metadata and derived annotations.
    """

    record = {}

    # Open FITS image of first filter (each should have same shape)
    with fits.open(images[FILT_INX], memmap=False, lazy_load_hdus=False) as hdul:
        height, width = hdul[0].data.shape

    # Open the FITS mask image
    with fits.open(mask, memmap=False, lazy_load_hdus=False) as hdul:
        hdul = hdul[1:]
        sources = len(hdul)
        # Normalize data
        data = [hdu.data for hdu in hdul]
        category_ids = [0 for hdu in hdul]

        ellipse_pars = [hdu.header["ELL_PARM"] for hdu in hdul]
        bbox = [list(map(int, hdu.header["BBOX"].split(","))) for hdu in hdul]

    # Add image metadata to record (should be the same for each filter)
    for f in filters:
        record[f"filename_{f.upper()}"] = images[filters.index(f)]

    # Assign file_name
    record[f"file_name"] = images[FILT_INX]
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width
    objs = []

    # Generate segmentation masks from model
    for i in range(sources):
        image = data[i]
        # Why do we need this?
        if len(image.shape) != 2:
            continue
        # Create mask from threshold
        mask = data[i]
        # Smooth mask
        # mask = cv2.GaussianBlur(mask, (9,9), 2)
        x, y, w, h = bbox[i]  # (x0, y0, w, h)

        # https://github.com/facebookresearch/Detectron/issues/100
        contours, _ = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            # contour = [x1, y1, ..., xn, yn]
            contour = contour.flatten()
            if len(contour) > 4:
                contour[::2] += x - w // 2
                contour[1::2] += y - h // 2
                segmentation.append(contour.tolist())
        # No valid contours
        if len(segmentation) == 0:
            continue

        # Add to dict
        obj = {
            # the scripts that run scarlet saves the center of the bounding box,
            # so we transform from center to bottom left.
            "bbox": [x - w // 2, y - h // 2, w, h],
            "area": w * h,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": segmentation,
            "category_id": category_ids[i],
            "ellipse_pars": ellipse_pars[i],
        }
        objs.append(obj)

    record["annotations"] = objs
    return record


def annotate_hsc_new(images, mask, idx, filters):
    """
    This can needs to be customized to your training data format

    """

    record = {}

    # Open FITS image of first filter (each should have same shape)
    with fits.open(images[FILT_INX], memmap=False, lazy_load_hdus=False) as hdul:
        height, width = hdul[0].data.shape

    # Open each FITS mask image
    print(mask)
    with fits.open(mask, memmap=False, lazy_load_hdus=False) as hdul:
        hdul = hdul[1:]
        sources = len(hdul)
        # Normalize data
        data = [hdu.data for hdu in hdul]
        try:
            category_ids = [hdu.header['c_id'] for hdu in hdul]
        except:
            category_ids = [0 for hdu in hdul]

        # ellipse_pars = [hdu.header["ELL_PARM"] for hdu in hdul]
        bbox = [list(map(int, hdu.header["BBOX"].split(","))) for hdu in hdul]
        area = [hdu.header["AREA"] for hdu in hdul]
            
        obj_ids = [hdu.header["objid"] for hdu in hdul]
        et_1 = [hdu.header["et_1"] for hdu in hdul]
        et_2 = [hdu.header["et_2"] for hdu in hdul]
        e_weight = [hdu.header["e_weight"] for hdu in hdul]
        e_rms = [hdu.header["e_rms"] for hdu in hdul]
        e_sigma = [hdu.header["e_sigma"] for hdu in hdul]
        has_shape = [hdu.header["has_e"] for hdu in hdul]

    catalog = images[FILT_INX].split(os.sep)[-6]
    catagory = images[FILT_INX].split(os.sep)[-5]
    tract = images[FILT_INX].split(os.sep)[-4]
    patch = images[FILT_INX].split(os.sep)[-3]
    sp = images[FILT_INX].split(os.sep)[-2]
    #patch = (
    #    int(bn.split("_")[2].split("_")[2][0]),
    #    int(bn.split("_")[2].split("_")[2][-1]),
    #)
    #patch = bn.split('_')[2]
    #sp = int(bn.split("_")[3])
    record[f"filename"] = f"/home/wenyinli/wl_deepdisc/datasets/{catalog}/{catagory}/{tract}/{patch}/{sp}/image"
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width
    objs = []

    # Generate segmentation masks from model
    for i in range(sources):
        image = data[i]
        # Why do we need this?
        if len(image.shape) != 2:
            continue
        height_mask, width_mask = image.shape
        # Create mask from threshold
        mask = data[i]
        # Smooth mask
        # mask = cv2.GaussianBlur(mask, (9,9), 2)
        x, y, w, h = bbox[i]  # (x0, y0, w, h)

        # https://github.com/facebookresearch/Detectron/issues/100
        contours, hierarchy = cv2.findContours(
            (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        segmentation = []
        for contour in contours:
            # contour = [x1, y1, ..., xn, yn]
            contour = contour.flatten()
            if len(contour) > 4:
                contour[::2] += x - w // 2
                contour[1::2] += y - h // 2
                segmentation.append(contour.tolist())
        # No valid countors
        if len(segmentation) == 0:
            print(i)
            continue

        # Add to dict
        obj = {
            "bbox": [x - w // 2, y - h // 2, w, h],
            "area": w * h,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": segmentation,
            "category_id": category_ids[i],
            # "ellipse_pars": ellipse_pars[i],
            "obj_id": obj_ids[i],
            "et_1": et_1[i],
            "et_2": et_2[i],
            "e_weight ": e_weight[i],
            "e_rms": e_rms[i],
            "e_sigma": e_sigma[i],
            "has_shape": has_shape[i],
            #"psfs": psfs[:,i],
        }

        objs.append(obj)

    record["annotations"] = objs

    return record