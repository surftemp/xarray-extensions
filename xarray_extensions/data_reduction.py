"""
Defines an xarray extensions for data reduction.

Importing this module attaches extra methods to xarray DataArray as follows:

xarray.DataArray:
    som - perform PCA along a particular dimension and return the new map coordinates as a new DataArray
    pca - perform PCA along a particular dimension and return the components as a new DataArray
"""

import random
import sys
import xarray as xr
import logging
import numpy as np

sklearn_available = False
try:
    from sklearn.decomposition import PCA
    sklearn_available = True
except:
    logging.warning("scikit-learn is not installed - some methods will not be available")


def pca(self, dimension="time", n_components=2, model_callback=None):
    """
    Extract and return the principle components for this data array after reducing along a target dimension

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound
    dimension: str
       the name of the target dimension to be reduced
    n_components: int
       the number of components to extract, which must be less than the length of the target dimension
    model_callback: function
       if provided, this method will be invoked with a sklearn.decomposition.PCA object after fitting
       (see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance with the target dimension replaced by a dimension of a size
       controlled by the n_components parameter

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    stack_dims = tuple(x for x in self.dims if x != dimension)
    stack_sizes = tuple(self.values.shape[i] for i in range(len(self.dims)) if self.dims[i] != dimension)

    instances = self.stack(case=stack_dims).transpose("case", dimension).values
    (nr_instances, nr_values) = instances.shape

    # for any instances containing at least one NaNs, record those and replace all values with dummy (zero) values.
    # PCA cannot handle NaNs.
    mask = np.all(~np.isnan(instances),axis=1)
    instance_mask = np.expand_dims(mask,1)
    instance_mask = np.tile(instance_mask,nr_values)
    instances = np.where(instance_mask,instances,0)

    pca = PCA(n_components=n_components)
    factor_scores = pca.fit_transform(instances)

    # for instances that had NaNs replaced with zero, replace the dummy factor scores with NaN.
    factor_scores_mask = np.expand_dims(mask, 1)
    factor_scores_mask = np.tile(factor_scores_mask, n_components)
    factor_scores = np.where(factor_scores_mask, factor_scores, np.nan)

    # reshape the factor scores back to the original axes
    a = factor_scores.reshape(stack_sizes + (n_components,))

    # create a DataArray object for output
    new_coords = {}
    for key in self.coords:
        if key != dimension:
            new_coords[key] = self.coords[key]
    new_dims = stack_dims+("pca_component",)
    if model_callback:
        model_callback(pca)
    return xr.DataArray(data=a, dims=new_dims, coords=new_coords)


from xarray_extensions.utils import bind_method

if sklearn_available:
    bind_method(xr.DataArray, pca)

