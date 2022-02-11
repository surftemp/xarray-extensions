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
import math
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

class MisalignedTimeAxisException(Exception):

    def __init__(self):
        super().__init__("Time axes for both arrays must be identical")

def __check_time_axis(da1, da2):
    tvals1 = da1.time.values
    tvals2 = da2.time.values
    if len(tvals1) != len(tvals2):
        raise MisalignedTimeAxisException()
    for i in range(len(tvals1)):
        if tvals2[i] != tvals1[i]:
            raise MisalignedTimeAxisException()

def __dof_sig_lvl(dof=5, C_I=0.05):
    t_Dist = stats.t.ppf(1 - C_I * 0.5, dof)
    V = dof - 2
    sq_t_Dist = math.pow(t_Dist, 2)
    r_sig = math.sqrt(sq_t_Dist / (V + sq_t_Dist))
    return r_sig

def __prepare_lagged_results(da,lags,extra_dim=None):
    # create a new data array to hold the results of lagged correlation or regression
    # in the result, the time dimension is replaced by a dimension to hold the lags
    # if an extra dimension is needed, supply a pair indicating its (name,size) as the extra_dim parameter
    time_index = da.dims.index("time")

    # replace the time dimension in the input data with lag
    newshape = list(da.shape)
    newdims = ["lag" if i == time_index else da.dims[i] for i in range(len(da.dims))]

    if extra_dim:
        (name,sz) = extra_dim
        newdims += [name]
        newshape += [sz]

    newshape[time_index] = len(lags)
    arr = np.zeros(tuple(newshape))
    coords = {"lag": lags}
    for coord in da.coords:
        if coord != "time" and coord in newdims:
                coords[coord] = da.coords[coord]

    return xr.DataArray(data=arr, dims=newdims, coords=coords)

def __compute_correlation(da,otherda,lag,idx,ci,dof,result):
    # compute correlation where self is shifted lag steps behind of otherda
    # store the correlation in the result array at lag=idx
    da_shift = da.copy(deep=True).shift({"time": lag})
    corr = xr.corr(da_shift, otherda, "time")
    # assign for this slice
    if ci:
        result.isel(lag=idx, parameter=0)[...] = corr
        # TODO work out degrees of freedom if not explicitly provided
        result.isel(lag=idx, parameter=1)[...] = __dof_sig_lvl(dof,ci) if dof else np.nan
    else:
        result.isel(lag=idx)[...] = corr

def lagged_correlation_month_of_year(self, otherda, lags, month_of_year, ci=None, dof=None):
    """
    Obtain pearson correlation coefficients between a yearly timeseries extracted from this DataArray at a particular month
    and those from another DataArray, with a series of lags applied return the correlation coefficients

    if significance threshold is to be calculated, return an extra parameter dimension with the correlation at index 0 and
    the two-tailed significance threshold at index 1

    Parameters
    ----------
    self: xarray.DataArray
       the main DataArray instance (to which this method is bound), assumed to include a "time" dimension
    otherda: xarray.DataArray
       the other DataArray against which the correlation is to be performed, assumed to have a "time" dimensions
    lags: list[int]
       a list of lags (in terms of numbers of months) to apply to the other dataset before calculating the correlation
       coefficient.  Note a lag of 1 means that the main data array lags a month after the other dataset.
    month_of_year: int
       indicate which month (1=jan, 2=feb, etc) to analyse in the main DataArray
    ci: float
       specify the confidence interval if significance is to be calculated (for example, specify 0.05 for 95% threshold)
    dof: int
       set the degrees of freedom manually
       (TODO, if not specified this should be computed from the data, currently return NaN)

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
       dimension

    Raises
    ------
    MisalignedTimeAxisException
       if this array and the other array do not have identical time coordinates

    Notes
    -----
    This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
    is imported.

    Example
    -------
    A way you might use me is::

        sla = ... get the monthly SLA as a DataArray
        sst = ... get the monthly SST as a DataArray

        # get the correlations between april SLAs and march SSTs over successive years
        correlation = sla.lagged_correlation_month_of_year(sst, lags=[-1], month_of_year=4)
    """

    __check_time_axis(self, otherda)

    result = __prepare_lagged_results(self, lags, ("parameter", 2) if ci else None)

    # select out the chosen month from the first array
    da0 = self.sel(time=self.time.dt.month.isin([month_of_year]))
    da0_times = list(map(lambda dt: dt.astype('datetime64[M]'), da0.time.values))
    da0 = da0.assign_coords(time=da0_times)

    for idx in range(len(lags)):
        lag = lags[idx]
        other_shift = otherda.shift({"time": lag})
        da1 = other_shift.sel(time=other_shift.time.dt.month.isin([month_of_year])).assign_coords(time=da0_times)
        __compute_correlation(da0, da1, 0, idx, ci, dof, result)

    return result

def lagged_correlation(self, otherda, lags, ci=None, dof=None):
    """
    Obtain pearson correlation coefficients between this DataArray and another DataArray, with a series of lags applied
    return the correlation coefficients

    if significance threshold is to be calculated, return an extra parameter dimension with the correlation at index 0 and
    the two-tailed significance threshold at index 1

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound, assumed to include a "time" dimension
    otherda: xarray.DataArray
       the other DataArray against which the correlation is to be performed, assumed to have a "time" dimensions
    lags: list[int]
       a list of lags to apply to the other dataset before calculating the correlation coefficient
       Note a lag of 1 means that the main data array lags a month after the other dataset.
    ci: float
       specify the confidence interval if significance is to be calculated (for example, specify 0.05 for 95% threshold)
    dof: int
       set the degrees of freedom manually
       (TODO, if not specified this should be computed from the data, currently return NaN)

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
       dimension

    Raises
    ------
    MisalignedTimeAxisException
       if this array and the other array do not have identical time coordinates

    Notes
    -----
    This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
    is imported.

    Example
    -------
    A way you might use me is::

        sla = ... get the monthly SLA as a DataArray
        sst = ... get the monthly SST as a DataArray

        # get the correlations between SLAs and SSTs that occur one month earlier
        correlation = sla.lagged_correlation(sst, lags=[-1])
    """

    __check_time_axis(self, otherda)

    result = __prepare_lagged_results(self, lags, ("parameter",2) if ci else None)

    for idx in range(len(lags)):
        lag = lags[idx]
        __compute_correlation(self,otherda,lag,idx,ci,dof,result)

    return result

def __compute_regression(da,otherda,lag,idx,result):

    def linregression(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        if not np.any(mask):
            return np.array([np.nan, np.nan])
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            return np.array([slope, intercept])

    # compute regression where da is shifted lag steps behind of otherda
    da_shift = da.shift({"time": lag})
    coeffs = xr.apply_ufunc(linregression, da_shift, otherda,
        input_core_dims=[['time'], ['time']], output_core_dims=[["parameter"]],
        vectorize=True, dask="parallelized", output_dtypes=['float64'])

     # assign for this slice
    result.isel(lag=idx)[...] = coeffs


def lagged_regression_month_of_year(self, otherda, lags, month_of_year):
    """
    Obtain linear regression coefficients between this between a yearly timeseries extracted from this DataArray
    at a particular month and those from another DataArray, with a series of lags applied, and return the
    regression coefficients.

    The other DataArray is treated as the y variable, this DataArray is treated as the x variable and the coefficients
    returned are the values [m,c] from y = mx+c

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound, assumed to include a "time" dimension
    otherda: xarray.DataArray
       the other DataArray against which the correlation is to be performed, assumed to have dimensions (time)
    lags: list[int]
       a list of lags to apply to the main dataset before calculating the regression coefficient for each lag
       Note a lag of 1 means that the main data array lags a month after the other dataset.

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
       dimension and an extra parameter dimension added (with size 2, where index 0 holds the slope value m and index 1
       holds the intercept value c)

    Raises
    ------
    MisalignedTimeAxisException
       if this array and the other array do not have identical time coordinates

    Notes
    -----
    This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
    is imported.

    Example
    -------
    A way you might use me is::

        sla = ... get the monthly SLA as a DataArray
        sst = ... get the monthly SST as a DataArray

        # get the regression coefficients for linear model relating June SLA from May and April SSTs
        # the returned coefficient parameters  m,c are such that sst = m*sla + c
        correlation = sst.lagged_regression_month_of_year(sla, lags=[1,2], month_of_year=6)
    """

    __check_time_axis(self, otherda)

    result = __prepare_lagged_results(self, lags, ("parameter", 2))

    # select out the chosen month from the first array
    da0 = self.sel(time=self.time.dt.month.isin([month_of_year]))
    da0_times = list(map(lambda dt: dt.astype('datetime64[M]'),da0.time.values))
    da0 = da0.assign_coords(time=da0_times)

    for idx in range(len(lags)):
        lag = lags[idx]
        other_shift = otherda.shift({"time": -lag})
        da1 = other_shift.sel(time=other_shift.time.dt.month.isin([month_of_year])).assign_coords(time=da0_times)
        __compute_regression(da0, da1, 0, idx, result)

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
       Note a lag of 1 means that the main data array lags a month after the other dataset.

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance having the same dimensions but with the time dimension replaced with the lags
       dimension and an extra parameter dimension added (with size 2, where index 0 holds the slope value m and index 1
       holds the intercept value c)

    Raises
    ------
    MisalignedTimeAxisException
       if this array and the other array do not have identical time coordinates

    Notes
    -----
    This function is attached to the DataArray class as a method when this module (xarray_extensions.timeseries)
    is imported.

    Example
    -------
    A way you might use me is::

        sla = ... get the monthly SLA as a DataArray
        sst = ... get the monthly SST as a DataArray

        # get the regression coefficients to fit a linear model predicting SLA from SSTs that occur one month and two months earlier
        # the returned coefficient parameters  m,c are such that sla = m*sst + c
        correlation = sst.lagged_regression(sla, lags=[1,2])
    """

    __check_time_axis(self, otherda)

    result = __prepare_lagged_results(self,lags,("parameter",2))

    for idx in range(len(lags)):
        lag = lags[idx]
        __compute_regression(self,otherda,lag,idx,result)

    return result


from .utils import bind_method

bind_method(xr.DataArray, deseasonalised)
bind_method(xr.DataArray, detrended)
bind_method(xr.DataArray, lagged_correlation)
bind_method(xr.DataArray, lagged_correlation_month_of_year)
bind_method(xr.DataArray, lagged_regression)
bind_method(xr.DataArray, lagged_regression_month_of_year)

if __name__ == '__main__':
    nlats = 1
    nlons = 1
    ntimes = 48
    import random
    import datetime
    import matplotlib.pyplot as plt
    rng = random.Random()
    da = xr.DataArray(data=np.array(
        [[[rng.random() for i in range(1, ntimes+1)] for lon in range(nlons)]
         for lat in range(nlats)]),
        dims=["lat", "lon", "time"],
        coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                         range(1, ntimes+1)],
                "lat": [lat for lat in range(nlats)],
                "lon": [lon for lon in range(nlons)]})

    da3 = da.lagged_correlation_month_of_year(da, lags=[1], month_of_year=1)
    print(da3)




