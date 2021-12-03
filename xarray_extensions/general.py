"""
Defines some useful Xarray extensions.

Importing this module attaches extra methods to xarray DataArray and Dataset classes as follows:

xarray.Dataset
    safe_assign - assign a dataarray to the dataset without losing some attribute values due to a bug in xarray
"""

import xarray as xr

from .check_version import check_version
check_version()

def longitude_center_zero(self,longitude_name="lon"):
    """
    Center longitude axis on 0.  If the original longitude axis ranges from 0 <= lon < 360, modify the dataset so that the new one should range from -180 to 180.

    Parameters
    ----------
    self: xarray.Dataset
       the DataSet instance to be modified in place
    longitude_name: str
       the name of the longitude dimension

    Returns
    -------
    xarray.Dataset
       A new dataset with the longitude axis adjusted and data arrays shifted accordingly.  The original attributes of the
       longitude dimension are retained, but valid_min and valid_max are modified (if present)

    Raises
    ------
    Exception
        If the input longitude axis has values outside the range 0 <= lon < 360

    Notes
    -----

    Contributed by Ross Maidment
    """
    if self[longitude_name].min() < 0 or self[longitude_name].max() >= 360:
        raise Exception("longitude coordinate values cannot be < 0 or >= 360")
    newds = self.assign_coords(**{longitude_name:(((self[longitude_name] + 180) % 360) - 180)}).sortby(longitude_name)
    newds[longitude_name].attrs = self[longitude_name].attrs.copy()
    if "valid_min" in newds[longitude_name].attrs:
        newds[longitude_name].attrs["valid_min"] = newds[longitude_name].min()
    if "valid_max" in newds[longitude_name].attrs:
        newds[longitude_name].attrs["valid_max"] = newds[longitude_name].max()
    return newds


def safe_assign(self, da, name=""):
    """
    Attach a DataArray as a variable to a Dataset, working around a bug in xarray when using simple assignment
    See https://github.com/pydata/xarray/issues/2245 (dimension attributes are sometimes lost)

    Parameters
    ----------
    self: xarray.Dataset
       the DataSet instance to which the DataArray is to be assigned
    da: xarray.DataArray
       the DataArray to assign
    name: str
       the name of the variable to assign.  If empty, uses the name of the DataArray.

    Notes
    -----

    This function is attached to the DataSet class as a method when this module is imported
    """
    if name:
        da.name = name

    # save attribute values from any dimensions also used in the DataArray
    dim_attrs = {}
    for dim in self.dims:
        if dim in da.dims:
            dim_attrs[dim] = self[dim].attrs
    self[da.name] = da
    # restore attribute values that may have been lost
    for dim in dim_attrs:
        self[dim].attrs = dim_attrs[dim]

from .utils import bind_method

bind_method(xr.Dataset, safe_assign)
bind_method(xr.Dataset, longitude_center_zero)