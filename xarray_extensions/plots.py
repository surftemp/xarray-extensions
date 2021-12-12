import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def rgb_plot(self,dimension,rgb_dimensions=[0,1,None],fixed_rgb=(0.5,0.5,0.5),null_rgb=(0,0,0)):
    """
    Plot a 3D array relying on a selected dimension of size 1, 2 or 3, for example, an array returned from the
    data reduction operators som or pca

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound
    dimension: str
       the name of the target dimension used to derive colour
    rgb_dimensions: tuple[int,int,int]
       indexes into the dimension to use for colours red,green abd blue (or None to use a fixed colour)
    fixed_rgb: tuple(float,float,float)
       for colour dimensions which are not indexed (rgb_dimensions parameter), specify the value in the range 0-1
    null_rgb: tuple(float,float,float)
       plot nan/null values using this rgb colour (values in the range 0-1)

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    fns = [lambda v:fixed_rgb[0],lambda v: fixed_rgb[1],lambda v: fixed_rgb[2]]

    def make_function(idx, dim_min,dim_max):
        return lambda v: (v[idx] - dim_min) / (dim_max - dim_min) if not np.isnan(v[idx]) else null_rgb[idx]

    for idx in range(len(rgb_dimensions)):
        dim = rgb_dimensions[idx]
        if dim is not None:
            dim_min = float(self[:,:,dim].min())
            dim_max = float(self[:,:,dim].max())
            if dim_max > dim_min:
                fns[idx] = make_function(idx,dim_min,dim_max)


    def colorize(v):
        r = fns[0](v)
        g = fns[1](v)
        b = fns[2](v)
        return np.array([r,g,b])

    rgb_data = xr.apply_ufunc(colorize, self,
        input_core_dims = [[dimension]], output_core_dims = [["rgb"]],
        vectorize = True, output_dtypes = ['float64'])

    plt.imshow(rgb_data)
    plt.grid(False)
    plt.show()

from xarray_extensions.utils import bind_method

bind_method(xr.DataArray, rgb_plot)
