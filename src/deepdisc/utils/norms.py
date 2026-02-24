from astropy.visualization import AsinhStretch, ImageNormalize
import numpy as np

def asinh_norm(data, vmin=None, vmax=None, a=0.1):
    finite_data = data[np.isfinite(data)] # ensure no NaN or inf vals
    if len(finite_data) == 0:
        return ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(a=a))
    # percentile-based limits
    if vmin is None and vmax is None:
        vmin, vmax = np.percentile(finite_data, [1, 99])
    else:
        return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=a))
    
def stretch_color(data, clipPercent):
    return np.percentile(data, (0 + clipPercent, 100 - clipPercent))