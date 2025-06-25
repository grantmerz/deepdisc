import numpy as np
import os
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import scarlet
import pandas as pd
from scipy.ndimage import zoom

from scarlet.display import AsinhMapping
stretch = 1
Q = 5
NORM = AsinhMapping(minimum=0, stretch=stretch, Q=Q)


def get_psf_itpl(tract, patch, n_blocks, sp, band):
    psf_fname = f'/home/wenyinli/wl_deepdisc/datasets/psf_data_25/{tract}/{patch}/{band}_psf_image.fits'
    with fits.open(psf_fname) as hdul_psf:
        psf_sam = hdul_psf[1].data
        wcs = WCS(hdul_psf[1].header).dropaxis(2)

    low_res_block_size = [psf_sam.shape[1] // n_blocks, psf_sam.shape[2] // n_blocks]
    centers = get_centers(low_res_block_size[::-1], n_blocks)
    coord = centers[sp]

    cutout_low_res = np.zeros([3, low_res_block_size[0], low_res_block_size[1]])
    for j in range(3):
        cutout_low_res[j] = Cutout2D(psf_sam[j], position=coord, size=low_res_block_size, wcs=wcs).data

    final_block_size = int(4200 // n_blocks)
    zoom_factor = [1, final_block_size / low_res_block_size[0], final_block_size / low_res_block_size[1]]
    
    return zoom(cutout_low_res, zoom_factor, order=1)
            
def get_psf(tract, patch, n_blocks, sp, catalog):
    filters = ['u','g','r','i','z','y']
    n_truth = len(catalog['new_x'].values)
    psf_gt = np.zeros([len(filters), 3, n_truth])
    for (i, band) in enumerate(filters):
        cutout = get_psf_itpl(tract, patch, n_blocks, sp, band)
        psf_gt[i] = np.array([cutout[:,round(pos[1]), round(pos[0])] for pos in catalog[['new_x','new_y']].values]).transpose()
    return psf_gt

def get_DC2_data(dirpath, filters=['u','g','r','i','z','y'], tract=10054, patch=[0,0], coord=None, cutout_size=[128, 128]):
    """
    Get HSC data given tract/patch info or SkyCoord
    
    Parameters
    ----------
    dirpath : str
        Path to HSC image file directory
    filters : list 
        A list of filters for your images. Default is ['g', 'r', 'i'].
    tract  : int
        An integer used for specifying the tract. Default is 10054|
    patch : [int, int]
        Patch #,#. Default is [0,0]
    coord  : SkyCoord
        Astropy SkyCoord, when specified, overrides tract/patch info and attempts to lookup HSC filename from ra, dec. 
        Default is None
    cutout_size: [int, int]
        Size of cutout to use (set to None for no cutting). Default is [128, 128]
        
    The image filepath is in the form:
        {dirpath}/deepCoadd/HSC-{filter}/{tract}/{patch[0]},{patch[1]}/calexp-HSC-{filter}-{tract}-{patch[0]},{patch[1]}.fits
    
    Returns
    -------
    data : ndarray
        HSC data array with dimensions [filters, N, N]
    """

    
    datas = []

    for f in filters:
        filepath = os.path.join('/',*[dirpath,f,tract,patch,f'calexp-{f}-{tract}-{patch}.fits'])
        
        #print(f'Loading "{filepath}".')
        #try:
        
        with fits.open(filepath) as obs_hdul:
        #obs_hdul = fits.open(filepath)
            data = obs_hdul[1].data
            wcs = WCS(obs_hdul[1].header)
        
        cutout =None
        
        # Cutout data at center of patch (coord=None) or at coord (if specified)
        if cutout_size is not None:
            # Use coord for center position if specified
            if coord is None:
                shape = np.shape(data)
                position = (shape[0]/2, shape[1]/2)
            else:
                position = coord
            #data = Cutout2D(data, position=position, size=cutout_size, wcs=wcs).data
            cutout = Cutout2D(data, position=position, size=cutout_size, wcs=wcs)
            data = cutout.data

        datas.append(data)
        #except:
        #    print('Missing filter ', f)
            

    return np.array(datas), cutout

def get_HSC(dirpath, filters=['G','R','I','Z','Y'], tract=10054, patch=[0,0], coord=None, cutout_size=[128, 128],
                           get_psf=False):
    """
    Get HSC data given tract/patch info or SkyCoord
    
    Parameters
    ----------
    dirpath : str
        Path to HSC image file directory
    filters : list 
        A list of filters for your images. Default is ['g', 'r', 'i'].
    tract  : int
        An integer used for specifying the tract. Default is 10054|
    patch : [int, int]
        Patch #,#. Default is [0,0]
    coord  : SkyCoord
        Astropy SkyCoord, when specified, overrides tract/patch info and attempts to lookup HSC filename from ra, dec. 
        Default is None
    cutout_size: [int, int]
        Size of cutout to use (set to None for no cutting). Default is [128, 128]
        
    The image filepath is in the form:
        {dirpath}/deepCoadd/HSC-{filter}/{tract}/{patch[0]},{patch[1]}/calexp-HSC-{filter}-{tract}-{patch[0]},{patch[1]}.fits
    
    Returns
    -------
    data : ndarray
        DC2 data array with dimensions [filters, N, N]
    """

    
    datas = []
    alldata = []
    wcs_s = []

    psf=None

    for band in filters:
        filepath = os.path.join(dirpath,str(tract)+'/'+str(patch)+'/calexp-HSC-'+band+'-'+str(tract)+'-'+str(patch)+'.fits')
        with fits.open(filepath) as obs_hdul:
            #obs_hdul = fits.open(filepath)
            alldata.append(obs_hdul[1].data)
            wcs_s.append(WCS(obs_hdul[1].header))
            if get_psf:
                psf = obs_hdul[3].data

        cutout =None
    
    for i,f in enumerate(filters):
        datai = alldata[i]
        # Cutout data at center of patch (coord=None) or at coord (if specified)
        if cutout_size is not None:
            # Use coord for center position if specified
            if coord is None:
                shape = np.shape(data)
                position = (shape[0]/2, shape[1]/2)
            else:
                position = coord
            #data = Cutout2D(data, position=position, size=cutout_size, wcs=wcs).data
            cutout = Cutout2D(datai, position=position, size=cutout_size, wcs=wcs_s[i])
            datai = cutout.data

        datas.append(datai)
            #except:
            #    print('Missing filter ', f)        


    return np.array(datas), cutout, psf

def get_DC2_data_alltracts(dirpath, filters=['u','g','r','i','z','y'], tract=10054, patch=[0,0], coord=None, cutout_size=[128, 128],
                           get_psf=False):
    """
    Get HSC data given tract/patch info or SkyCoord
    
    Parameters
    ----------
    dirpath : str
        Path to HSC image file directory
    filters : list 
        A list of filters for your images. Default is ['g', 'r', 'i'].
    tract  : int
        An integer used for specifying the tract. Default is 10054|
    patch : [int, int]
        Patch #,#. Default is [0,0]
    coord  : SkyCoord
        Astropy SkyCoord, when specified, overrides tract/patch info and attempts to lookup HSC filename from ra, dec. 
        Default is None
    cutout_size: [int, int]
        Size of cutout to use (set to None for no cutting). Default is [128, 128]
        
    The image filepath is in the form:
        {dirpath}/deepCoadd/HSC-{filter}/{tract}/{patch[0]},{patch[1]}/calexp-HSC-{filter}-{tract}-{patch[0]},{patch[1]}.fits
    
    Returns
    -------
    data : ndarray
        DC2 data array with dimensions [filters, N, N]
    """

    
    datas = []
    filepath = os.path.join(dirpath,f'{tract}_{patch}_images.fits')

    psf=None

    
    with fits.open(filepath) as obs_hdul:
        #obs_hdul = fits.open(filepath)
        alldata = obs_hdul[1].data
        wcs = WCS(obs_hdul[1].header).dropaxis(2)
        if get_psf:
            psf = obs_hdul[2].data
        
    cutout =None
    
    for i,f in enumerate(filters):
        datai = alldata[i]
        # Cutout data at center of patch (coord=None) or at coord (if specified)
        if cutout_size is not None:
            # Use coord for center position if specified
            if coord is None:
                shape = np.shape(data)
                position = (shape[0]/2, shape[1]/2)
            else:
                position = coord
            #data = Cutout2D(data, position=position, size=cutout_size, wcs=wcs).data
            cutout = Cutout2D(datai, position=position, size=cutout_size, wcs=wcs)
            datai = cutout.data

        datas.append(datai)
            #except:
            #    print('Missing filter ', f)        


    return np.array(datas), cutout, psf

def get_DC2_psf_alltracts(dirpath, filters=['u','g','r','i','z','y'], tract=10054, patch=[0,0], coord=None, cutout_size=[128, 128], get_psf=False):
    """
    Get HSC data given tract/patch info or SkyCoord
    
    Parameters
    ----------
    dirpath : str
        Path to HSC image file directory
    filters : list 
        A list of filters for your images. Default is ['g', 'r', 'i'].
    tract  : int
        An integer used for specifying the tract. Default is 10054|
    patch : [int, int]
        Patch #,#. Default is [0,0]
    coord  : SkyCoord
        Astropy SkyCoord, when specified, overrides tract/patch info and attempts to lookup HSC filename from ra, dec. 
        Default is None
    cutout_size: [int, int]
        Size of cutout to use (set to None for no cutting). Default is [128, 128]
        
    The image filepath is in the form:
        {dirpath}/deepCoadd/HSC-{filter}/{tract}/{patch[0]},{patch[1]}/calexp-HSC-{filter}-{tract}-{patch[0]},{patch[1]}.fits
    
    Returns
    -------
    data : ndarray
        HSC data array with dimensions [filters, N, N]
    """

    
    datas = []
    dirpath = '/home/wenyinli/wl_deepdisc/datasets/CosmoDC2/psf_img/'
    psf=None
    cutout =None
    for i,f in enumerate(filters):
        filepath = os.path.join(dirpath+str(tract)+'/'+str(patch)+'/'+f+'_psf_image.fits')
        with fits.open(filepath) as obs_hdul:
            #obs_hdul = fits.open(filepath)
            alldata = obs_hdul[1].data
            wcs = WCS(obs_hdul[1].header)
            if get_psf:
                psf = obs_hdul[2].data
        datai = alldata
        # Cutout data at center of patch (coord=None) or at coord (if specified)
        if cutout_size is not None:
            # Use coord for center position if specified
            if coord is None:
                shape = np.shape(data)
                position = (shape[0]/2, shape[1]/2)
            else:
                position = coord
            #data = Cutout2D(data, position=position, size=cutout_size, wcs=wcs).data
            cutout = Cutout2D(datai, position=position, size=cutout_size, wcs=wcs)
            datai = cutout.data

        datas.append(datai)
            #except:
            #    print('Missing filter ', f)        


    return np.array(datas), cutout, psf

def get_centers(sub_shape,n):
    centers=[]
    for i in range(n):
        for j in range(n):
            #print(sub_shape[1]*i)
            s=np.array(sub_shape)/2 + (sub_shape[0]*j, sub_shape[1]*i)
            centers.append(s)
            
    return centers


def get_cutout(dirpath,tract,patch,sp,nblocks=4,filters=['u','g','r','i','z','y'],plot=False, get_psf=True):

    #dat,cutout = get_DC2_data(dirpath,filters=filters,tract=tract,patch=patch,coord=None,cutout_size=None)
    dat,cutout,psf = get_DC2_data_alltracts(dirpath,filters=filters,tract=tract,patch=patch,coord=None,cutout_size=None, get_psf=get_psf)

    
    block_size = [dat.shape[1]//nblocks, dat.shape[2]//nblocks]

    
    sub_shape =[dat.shape[1]//nblocks,dat.shape[2]//nblocks]
    centers = get_centers(sub_shape[::-1],nblocks)

    coord=centers[sp]

    #datsm,cutout = get_DC2_data(dirpath,tract=tract,patch=patch,coord=coord,cutout_size=sub_shape)
    datsm,cutout,psf = get_DC2_data_alltracts(dirpath,tract=tract,patch=patch,coord=coord,cutout_size=sub_shape, get_psf=get_psf)
    datsp, _ ,_ = get_DC2_psf_alltracts(dirpath,tract=tract,patch=patch,coord=coord,cutout_size=sub_shape, get_psf=get_psf)
    dats_all = np.concatenate((datsm, datsp), axis = 0)
    if plot:
        fig,ax = plt.subplots(1,2,figsize=(10,10))
        img_rgb = scarlet.display.img_to_rgb(dat, norm=NORM)
        img_rgbsm = scarlet.display.img_to_rgb(datsm, norm=NORM)

        ax[0].imshow(img_rgb,origin='lower')
        cutout.plot_on_original(ax[0],color='white')
        ax[1].imshow(img_rgbsm,origin='lower')

        ax[0].axis('off')
        ax[1].axis('off')
        plt.tight_layout()
    
    return cutout,dats_all, psf


def get_cutout_HSC(dirpath,tract,patch,sp,nblocks=4,filters=['G','R','I','Z','Y'],plot=False, get_psf=True):

    #dat,cutout = get_DC2_data(dirpath,filters=filters,tract=tract,patch=patch,coord=None,cutout_size=None)
    dat,cutout,psf = get_HSC(dirpath,filters=filters,tract=tract,patch=patch,coord=None,cutout_size=None, get_psf=get_psf)

    
    block_size = [dat.shape[1]//nblocks, dat.shape[2]//nblocks]

    
    sub_shape =[dat.shape[1]//nblocks,dat.shape[2]//nblocks]
    centers = get_centers(sub_shape[::-1],nblocks)

    coord=centers[sp]

    #datsm,cutout = get_DC2_data(dirpath,tract=tract,patch=patch,coord=coord,cutout_size=sub_shape)
    datsm,cutout,psf = get_HSC(dirpath,tract=tract,patch=patch,coord=coord,cutout_size=sub_shape, get_psf=get_psf)
    if plot:
        fig,ax = plt.subplots(1,2,figsize=(10,10))
        img_rgb = scarlet.display.img_to_rgb(dat, norm=NORM)
        img_rgbsm = scarlet.display.img_to_rgb(datsm, norm=NORM)

        ax[0].imshow(img_rgb,origin='lower')
        cutout.plot_on_original(ax[0],color='white')
        ax[1].imshow(img_rgbsm,origin='lower')

        ax[0].axis('off')
        ax[1].axis('off')
        plt.tight_layout()
    
    return cutout,datsm, psf

def get_cutout_cat(dirpath,dall,tract,patch,sp,nblocks=4,filters=['u','g','r','i','z','y']):
    '''
        WARNING!!!!!
        It is not efficient to have the full catalog as input when doing multiprocesing.  
        Keep it in the top level process
    '''

    cutout,datsm= get_cutout(tract=tract,patch=patch,sp=sp,nblocks=nblocks, filters=filters,plot=False)
    xs,ys = cutout.wcs.world_to_pixel(allcatalog)
    inds = np.where((xs>=0) & (xs<cutout.shape[1]-1) & (ys>=0) & (ys<cutout.shape[0]-1))[0]
    
    dcut = dall.iloc[inds]

    dcut['new_x'] = xs[inds]
    dcut['new_y'] = ys[inds]

    column_to_move = dcut.pop("objectId")

    # insert column with insert(location, column_name, column_value)
    dcut.insert(0, "objectId", column_to_move)
    dcut.sort_values(by='objectId')
    
    return datsm, dcut