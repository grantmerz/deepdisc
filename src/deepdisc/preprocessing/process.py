import numpy as np
import scarlet
import astropy.io.fits as fits
import os
import h5py
import pandas as pd


def write_scarlet_results(
    datas,
    observation,
    starlet_sources,
    model_frame,
    catalog_deblended,
    segmentation_masks,
    outdir,
    filters,
    s,
    catalog=None,
):
    """
    Saves images in each channel, with headers for each source in image,
    such that the number of headers = number of sources detected in image.

    Parameters
    ----------
    datas: array
        array of Data objects
    observation: scarlet function
        Scarlet observation objects
    starlet_sources: list
        List of ScarletSource objects
    model_frame: scarlet function
        Image frame of source model
    catalog_deblended: list
        Deblended source detection catalog
    catalog: pandas df
        External catalog of source detections
    segmentation_masks: list
        List of segmentation mask of each object in image
    outdir : str
        Path to HSC image file directory
    filters : list
        A list of filters for your images. Default is ['g', 'r', 'i'].
    s : str
        File basename string


    Returns
    -------
    filename : dict
        dictionary of all paths to the saved scarlet files for the particular dataset.
        Saved image and model files for each filter, and one total segmentation mask file for all filters.
    """

    def _make_hdr(starlet_source, cat, source_cat=None):
        """
        Helper function to make FITS header and insert metadata.
        Parameters
        ----------
        starlet_source: starlet_source
            starlet_source object for source k
        cat: dict
            catalog data for source k

        Returns
        -------
        model_hdr : Astropy fits.Header
            FITS header for source k with catalog metadata
        """
        # For each header, assign descriptive data about each source
        # (x0, y0, w, h) in absolute floating pixel coordinates
        bbox_h = starlet_source.bbox.shape[1]
        bbox_w = starlet_source.bbox.shape[2]
        bbox_y = starlet_source.bbox.origin[1] + int(np.floor(bbox_w / 2))  # y-coord of the source's center
        bbox_x = starlet_source.bbox.origin[2] + int(np.floor(bbox_w / 2))  # x-coord of the source's center

        
        # Add info to header
        model_hdr = fits.Header()
        model_hdr["bbox"] = ",".join(map(str, [bbox_x, bbox_y, bbox_w, bbox_h]))
        model_hdr["area"] = bbox_w * bbox_h

        if source_cat is not None:
            for key in source_cat.keys():
                value = source_cat[key]
                if not np.isfinite(value):
                    imag = -1
                model_hdr[key] = value
            
        else:
            # Ellipse parameters (a, b, theta) from deblend catalog
            e_a, e_b, e_theta = cat["a"], cat["b"], cat["theta"]
            ell_parm = np.concatenate((cat["a"], cat["b"], cat["theta"]))
            model_hdr["ell_parm"] = ",".join(map(str, list(ell_parm)))
            model_hdr["cat_id"] = 1  # Category ID



        return model_hdr

    # Create dict for all saved filenames
    segmask_hdul = []
    model_hdul = []
    filenames = {}

    # Filter loop
    for i, f in enumerate(filters):
        f = f.upper()

        # Primary HDU is full image
        img_hdu = fits.PrimaryHDU(data=datas[i])
        
        # Create header entry for each scarlet source
        for k, (src, cat) in enumerate(zip(starlet_sources, catalog_deblended)):
            if catalog is not None:
                source_cat = catalog.iloc[k]
            else:
                source_cat=None
            # Get each model, make into image
            model = starlet_sources[k].get_model(frame=model_frame)
            model = observation.render(model)
            model = src.bbox.extract_from(model)

            model_hdr = _make_hdr(starlet_sources[k], cat, source_cat)

            model_hdu = fits.ImageHDU(data=model[i], header=model_hdr)
            model_primary = fits.PrimaryHDU()

            model_hdul.append(model_hdu)

        # Write final fits file to specified location
        # Save full image and then headers per source w/ descriptive info
        save_img_hdul = fits.HDUList([img_hdu])
        save_model_hdul = fits.HDUList([model_primary, *model_hdul])

        # Save list of filenames in dict for each band
        #filenames["img"] = os.path.join(outdir, f"{s}_images.npy")
        #np.save(filenames["img"],datas)
        filenames[f"img_{f}"] = os.path.join(outdir, f"{f}_{s}_scarlet_img.fits")
        save_img_hdul.writeto(filenames[f"img_{f}"], overwrite=True)
        
        filenames[f"model_{f}"] = os.path.join(outdir, f"{f}_{s}_scarlet_model.fits")
        save_model_hdul.writeto(filenames[f"model_{f}"], overwrite=True)

    # If we have segmentation mask data, save them as a separate fits file
    # Just using the first band for the segmentation mask
    if segmentation_masks is not None:
        for i, f in enumerate(filters[0]):
            # Create header entry for each scarlet source
            for k, (src, cat) in enumerate(zip(starlet_sources, catalog_deblended)):
                if catalog is not None:
                    source_cat = catalog.iloc[k]
                else:
                    source_cat=None

                segmask_hdr = _make_hdr(starlet_sources[k], cat, source_cat)

                # Save each model source k in the image
                segmask_hdu = fits.ImageHDU(data=segmentation_masks[k], header=segmask_hdr)
                segmask_primary = fits.PrimaryHDU()

                segmask_hdul.append(segmask_hdu)

            save_segmask_hdul = fits.HDUList([segmask_primary, *segmask_hdul])

            # Save list of filenames in dict for each band
            filenames["segmask"] = os.path.join(outdir, f"{s}_scarlet_segmask.fits")
            save_segmask_hdul.writeto(filenames["segmask"], overwrite=True)

    return filenames





def write_scarlet_results_nomodels(
    datas,
    observation,
    starlet_sources,
    model_frame,
    segmentation_masks,
    outdir,
    filters,
    s,
    catalog=None,
):
    """
    Saves images in each channel, with headers for each source in image,
    such that the number of headers = number of sources detected in image.

    Parameters
    ----------
    datas: array
        array of Data objects
    observation: scarlet function
        Scarlet observation objects
    starlet_sources: list
        List of ScarletSource objects
    model_frame: scarlet function
        Image frame of source model
    catalog_deblended: list
        Deblended source detection catalog
    catalog: pandas df
        External catalog of source detections
    segmentation_masks: list
        List of segmentation mask of each object in image
    outdir : str
        Path to HSC image file directory
    filters : list
        A list of filters for your images. Default is ['g', 'r', 'i'].
    s : str
        File basename string


    Returns
    -------
    filename : dict
        dictionary of all paths to the saved scarlet files for the particular dataset.
        Saved image and model files for each filter, and one total segmentation mask file for all filters.
    """

    def _make_hdr(starlet_source, source_cat=None):
        """
        Helper function to make FITS header and insert metadata.
        Parameters
        ----------
        starlet_source: starlet_source
            starlet_source object for source k
        cat: dict
            catalog data for source k

        Returns
        -------
        model_hdr : Astropy fits.Header
            FITS header for source k with catalog metadata
        """
        # For each header, assign descriptive data about each source
        # (x0, y0, w, h) in absolute floating pixel coordinates
        bbox_h = starlet_source.bbox.shape[1]
        bbox_w = starlet_source.bbox.shape[2]
        bbox_y = starlet_source.bbox.origin[1] + int(np.floor(bbox_w / 2))  # y-coord of the source's center
        bbox_x = starlet_source.bbox.origin[2] + int(np.floor(bbox_w / 2))  # x-coord of the source's center

        
        # Add info to header
        model_hdr = fits.Header()
        model_hdr["bbox"] = ",".join(map(str, [bbox_x, bbox_y, bbox_w, bbox_h]))
        model_hdr["area"] = bbox_w * bbox_h

        if source_cat is not None:
            #catalog_redshift = source_cat["redshift_truth"]
            #oid = source_cat["objectId"]
            catalog_redshift = source_cat["redshift"]
            oid = source_cat["id"]
            imag = source_cat["mag_i"]
            shear_1 = source_cat["shear_1"]
            shear_2 = source_cat["shear_2"]
            convergence = source_cat["convergence"]
            et_1 = source_cat["ellipticity_1_true"]
            et_2 = source_cat["ellipticity_2_true"]
            size_1 = source_cat["size_true"]
            
            if not np.isfinite(imag):
                imag = -1
            #model_hdr["cat_id"] = source_cat['truth_type']  # Category ID
            model_hdr["redshift"] = catalog_redshift
            model_hdr["objid"] = oid
            model_hdr["mag_i"] = imag
            model_hdr["shear_1"] = shear_1
            model_hdr["shear_2"] = shear_2
            model_hdr["kappa"] = convergence
            model_hdr["et_1"] = et_1
            model_hdr["et_2"] = et_2
            model_hdr["size_1"] = size_1
            #for psf_i in range(18):
            #    model_hdr["psf_"+str(psf_i)] = source_cat["psf_"+str(psf_i)]

        return model_hdr

    # Create dict for all saved filenames
    segmask_hdul = []
    filenames = {}

    # Filter loop
    for i, f in enumerate(filters):
        #f = f.upper()

        # Primary HDU is full image
        img_hdu = fits.PrimaryHDU(data=datas[i])
        

        # Write final fits file to specified location
        # Save full image and then headers per source w/ descriptive info
        save_img_hdul = fits.HDUList([img_hdu])
        #save_model_hdul = fits.HDUList([model_primary, *model_hdul])

        # Save list of filenames in dict for each band
        #filenames["img"] = os.path.join(outdir, f"{s}_images.npy")
        #np.save(filenames["img"],datas)
        filenames[f"img_{f}"] = os.path.join(outdir, f"image_{f}.fits")
        save_img_hdul.writeto(filenames[f"img_{f}"], overwrite=True)
        
        #filenames[f"model_{f}"] = os.path.join(outdir, f"{f}_{s}_scarlet_model.fits")
        #save_model_hdul.writeto(filenames[f"model_{f}"], overwrite=True)

    # If we have segmentation mask data, save them as a separate fits file
    # Just using the first band for the segmentation mask
    if segmentation_masks is not None:
        for i, f in enumerate(filters[0]):
            # Create header entry for each scarlet source
            for k, src in enumerate(starlet_sources):
                if catalog is not None:
                    source_cat = catalog.iloc[k]
                else:
                    source_cat=None

                #segmask_hdr = _make_hdr(starlet_sources[k], cat, source_cat)
                segmask_hdr = _make_hdr(starlet_sources[k], source_cat)

                # Save each model source k in the image
                segmask_hdu = fits.ImageHDU(data=segmentation_masks[k], header=segmask_hdr)
                segmask_primary = fits.PrimaryHDU()

                segmask_hdul.append(segmask_hdu)

            save_segmask_hdul = fits.HDUList([segmask_primary, *segmask_hdul])

            # Save list of filenames in dict for each band
            filenames["segmask"] = os.path.join(outdir, "masks.fits")
            save_segmask_hdul.writeto(filenames["segmask"], overwrite=True)

    return filenames



def write_scarlet_results_HSC(
    datas,
    observation,
    starlet_sources,
    model_frame,
    segmentation_masks,
    outdir,
    filters,
    s,
    source_catalog=None,
):
    """
    Saves images in each channel, with headers for each source in image,
    such that the number of headers = number of sources detected in image.

    Parameters
    ----------
    datas: array
        array of Data objects
    observation: scarlet function
        Scarlet observation objects
    starlet_sources: list
        List of ScarletSource objects
    model_frame: scarlet function
        Image frame of source model
    catalog_deblended: list
        Deblended source detection catalog
    source_catalog: pandas df
        External catalog of source detections
    segmentation_masks: list
        List of segmentation mask of each object in image
    outdir : str
        Path to HSC image file directory
    filters : list
        A list of filters for your images. Default is ['g', 'r', 'i'].
    s : str
        File basename string


    Returns
    -------
    filename : dict
        dictionary of all paths to the saved scarlet files for the particular dataset.
        Saved image and model files for each filter, and one total segmentation mask file for all filters.
    """

    def _make_hdr(starlet_source, source_cat=None):
        """
        Helper function to make FITS header and insert metadata.
        Parameters
        ----------
        starlet_source: starlet_source
            starlet_source object for source k
        cat: dict
            catalog data for source k

        Returns
        -------
        model_hdr : Astropy fits.Header
            FITS header for source k with catalog metadata
        """
        # For each header, assign descriptive data about each source
        # (x0, y0, w, h) in absolute floating pixel coordinates
        bbox_h = starlet_source.bbox.shape[1]
        bbox_w = starlet_source.bbox.shape[2]
        bbox_y = starlet_source.bbox.origin[1] + int(np.floor(bbox_w / 2))  # y-coord of the source's center
        bbox_x = starlet_source.bbox.origin[2] + int(np.floor(bbox_w / 2))  # x-coord of the source's center

        
        # Add info to header
        model_hdr = fits.Header()
        model_hdr["bbox"] = ",".join(map(str, [bbox_x, bbox_y, bbox_w, bbox_h]))
        model_hdr["area"] = bbox_w * bbox_h

        if source_cat is not None:
            oid = source_cat["object_id"].astype(int)
            et_1 = source_cat["e1"]
            et_2 = source_cat["e2"]
            e_weight = source_cat['shape_weight']
            e_rms = source_cat['rms_e']
            e_sigma = source_cat['sigma_e']
            mag_i = source_cat['i_cmodel_mag']
            has_shape = source_cat['has_shape']
            
            if source_cat['i_calib_psf_used']:
                category_id = 1
            else:
                category_id = 0
                
            model_hdr["objid"] = oid
            model_hdr["et_1"] = et_1
            model_hdr["et_2"] = et_2
            model_hdr["e_weight"] = e_weight
            model_hdr["e_rms"] = e_rms
            model_hdr["e_sigma"] = e_sigma
            model_hdr["has_e"] = has_shape
            model_hdr["c_id"] = category_id
            model_hdr["mag_i"] = mag_i
            #for psf_i in range(18):
            #    model_hdr["psf_"+str(psf_i)] = source_cat["psf_"+str(psf_i)]

        return model_hdr

    # Create dict for all saved filenames
    segmask_hdul = []
    filenames = {}
    # Filter loop
    for i, f in enumerate(filters):
        #f = f.upper()

        # Primary HDU is full image
        img_hdu = fits.PrimaryHDU(data=datas[i])
        

        # Write final fits file to specified location
        # Save full image and then headers per source w/ descriptive info
        save_img_hdul = fits.HDUList([img_hdu])
        #save_model_hdul = fits.HDUList([model_primary, *model_hdul])

        # Save list of filenames in dict for each band
        #filenames["img"] = os.path.join(outdir, f"{s}_images.npy")
        #np.save(filenames["img"],datas)
        filenames[f"img_{f}"] = os.path.join(outdir, f"image_{f}.fits")
        save_img_hdul.writeto(filenames[f"img_{f}"], overwrite=True)
        
        #filenames[f"model_{f}"] = os.path.join(outdir, f"{f}_{s}_scarlet_model.fits")
        #save_model_hdul.writeto(filenames[f"model_{f}"], overwrite=True)

    # If we have segmentation mask data, save them as a separate fits file
    # Just using the first band for the segmentation mask
    if segmentation_masks is not None:
        for i, f in enumerate(filters[0]):
            # Create header entry for each scarlet source
            for k, src in enumerate(starlet_sources):
                if source_catalog is not None:
                    source_catalog = pd.DataFrame(source_catalog)
                    source_cat = source_catalog.iloc[k]
                else:
                    source_cat=None

                #segmask_hdr = _make_hdr(starlet_sources[k], cat, source_cat)
                segmask_hdr = _make_hdr(starlet_sources[k], source_cat)

                # Save each model source k in the image
                segmask_hdu = fits.ImageHDU(data=segmentation_masks[k], header=segmask_hdr)
                segmask_primary = fits.PrimaryHDU()

                segmask_hdul.append(segmask_hdu)

            save_segmask_hdul = fits.HDUList([segmask_primary, *segmask_hdul])

            # Save list of filenames in dict for each band
            filenames["segmask"] = os.path.join(outdir, "masks.fits")
            save_segmask_hdul.writeto(filenames["segmask"], overwrite=True)

    return filenames
