"""
Defines some useful Xarray extensions.

Importing this module attaches extra methods to xarray DataArray and Dataset classes as follows:

xarray.Dataset
    safe_assign - assign a dataarray to the dataset without losing some attribute values due to a bug in xarray
"""

import xarray as xr

from .check_version import check_version
check_version()

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