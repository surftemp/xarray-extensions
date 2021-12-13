"""
Defines some useful Xarray extensions.

Importing this module attaches extra methods to xarray DataArray and Dataset classes as follows:

xarray.DataArray:
    deseasonalised - model monthly climatology and return model or remove seasonal pattern from data
    detrended - model trends using linear or quadratic and return model or remove trend from data
    lagged_correlation - compute and return the pearson correlation or regression coefficient
               with another dataarray for specified lags/leads

xarray.Dataset
    safe_assign - assign a dataarray to the dataset without losing some attribute values due to a bug in xarray
"""

from scipy import stats
import xarray as xr
import numpy as np

from .check_version import check_version
check_version()

def deseasonalised(self, clim=False, abs=True):
    """
    Obtain a deseasonalised DataArray based on a monthly climatology

    Parameters
    ----------
    self: xarray.DataArray
        the DataArray instance to which this method is bound
    clim: boolean
        if True, return the climatology
    abs: boolean
        if False, return monthly anomalies, otherwise return the anomalies plus the monthly means from the climatology

    Returns
    -------
    xarray.DataArray
        an xarray.DataArray instance

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    # get the seasonal means
    monthly_means = self.groupby("time.month").mean()  # (month,lat,lon)
    if clim:
        return monthly_means
    else:
        anomalies = self.groupby("time.month") - monthly_means
        if abs:
            return anomalies + monthly_means.mean(dim="month")
        else:
            return anomalies


def detrended(self, quadratic=False, coeff=False, coeff_scale="year", abs=False):
    """
    Obtain a detrended DataArray using a linear or quadratic function to fit a trend

    Parameters
    ----------
    self: xarray.DataArray
        the DataArray instance to which this method is bound
    quadratic: boolean
        if True, fit a quadratic (degree=2) model, otherwise fit a linear (degree=1) model, along the time axis
    coeff: boolean
        if True, return the model coefficients
    coeff_scale: str
        set coefficients to work on a particular timescale, should be one of "year", "day", "second".
        set to None to scale using nanoseconds (xarray's default representation)
    abs: boolean
        if False, return differences between the original values and the model values, otherwise return the differences
        plus the model means

    Returns
    -------
    xarray.DataArray
        an xarray.DataArray instance

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    degree = 2 if quadratic else 1
    coeffs = self.polyfit(dim="time", deg=degree)
    coeffs.load()

    if coeff:
        # work out a scale factor, by default coefficients are fitted using xarray's nanosecond time representation
        if coeff_scale == "year":
            scale_factor = 365.25 * 24 * 60 * 60 * 1e9
        elif coeff_scale == "day":
            scale_factor = 24 * 60 * 60 * 1e9
        elif coeff_scale == "second":
            scale_factor = 1e9
        elif coeff_scale is None:
            scale_factor = 1
        else:
            raise Exception("coeff_scale should be one of year,day,second, or None")

        coeffs = coeffs["polyfit_coefficients"].data
        # first dimension is the degree, highest power first
        if scale_factor != 1:
            if quadratic:
                coeffs[0,:,:] *= (scale_factor * scale_factor)
                coeffs[1,:,:] *= scale_factor
            else:
                coeffs[0,:,:] *= scale_factor

        degree_coordinates = [2, 1, 0] if quadratic else [1, 0]
        degree_dimension = "degree2" if quadratic else "degree1"

        # for some reason transposing the dimensions helps when assigning the DataArray into the original DataSet
        return xr.DataArray(data=coeffs.transpose([1, 2, 0]),
                            dims=["lat", "lon", degree_dimension],
                            coords={"lat": self.coords["lat"], "lon": self.coords["lon"],
                                    degree_dimension: degree_coordinates})
    else:
        trend = xr.polyval(self["time"], coeffs["polyfit_coefficients"])
        diffs = self - trend
        if abs:
            diffs = diffs + trend.mean(dim="time")
        return diffs

def lagged_correlation(self, otherda, lags):
    """
    Obtain pearson correlation coefficients between this DataArray and another DataArray, with a series of lags applied

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound, assumed to include a "time" dimension
    otherda: xarray.DataArray
       the other DataArray against which the correlation is to be performed, assumed to have dimensions (time)
    lags: list[int]
       a list of lags to apply to the other dataset before calculating the correlation coefficient
    coefficient_type: str
       the type of coefficient to compute, either "pearson" or "regression".
       if "regression", the regression model is trained to estimate values in "otherda" based on values this array
       and the slope of is returned.

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
       dimension

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    time_index = self.dims.index("time")

    # replace the time dimension in the input data with lag
    newshape = list(self.shape)
    newdims = ["lag" if i == time_index else self.dims[i] for i in range(len(self.dims))]
    newshape[time_index] = len(lags)
    arr = np.zeros(tuple(newshape))
    result = xr.DataArray(data=arr, dims=newdims, coords={"lag": lags})
    for idx in range(len(lags)):
        lag = lags[idx]
        # compute correlation where self is shifted lag steps ahead of otherda
        da_shift = self.copy(deep=True).shift({"time": lag})
        corr = xr.corr(da_shift, otherda, "time")
        # assign for this slice
        result.isel(lag=idx)[...] = corr
    return result

def lagged_regression(self, otherda, lags):
    """
    Obtain linear regression coefficients between this DataArray and another DataArray, with a series of lags applied.
    The other DataArray is treated as the y variable, this DataArray is treated as the x variable and the coefficients
    returned are the values [m,c] from y = mx+c

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound, assumed to include a "time" dimension
    otherda: xarray.DataArray
       the other DataArray against which the correlation is to be performed, assumed to have dimensions (time)
    lags: list[int]
       a list of lags to apply to the other dataset before calculating the regression coefficient for each lag

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
       dimension and an extra parameter dimension added (with size 2, where index 0 holds the slope value m and index 1
       holds the intercept value c)

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    time_index = self.dims.index("time")

    # replace the time dimension in the input data with lag
    newshape = list(self.shape) + [2]
    newdims = ["lag" if i == time_index else self.dims[i] for i in range(len(self.dims))] + ["parameter"]
    newshape[time_index] = len(lags)
    arr = np.zeros(tuple(newshape))
    result = xr.DataArray(data=arr, dims=newdims, coords={"lag": lags})

    def linregression(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        if not np.any(mask):
            return np.array([np.nan,np.nan])
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            return np.array([slope,intercept])

    for idx in range(len(lags)):
        lag = lags[idx]
        # compute correlation where self is shifted lag steps ahead of otherda
        da_shift = self.copy(deep=True).shift({"time": lag})
        corr = xr.apply_ufunc(linregression, da_shift, otherda,
            input_core_dims=[['time'], ['time']], output_core_dims=[["parameter"]],
            vectorize=True, dask="parallelized", output_dtypes=['float64'])
         # assign for this slice
        result.isel(lag=idx)[...] = corr
    return result


from .utils import bind_method

bind_method(xr.DataArray, deseasonalised)
bind_method(xr.DataArray, detrended)
bind_method(xr.DataArray, lagged_correlation)
bind_method(xr.DataArray, lagged_regression)


