import unittest
import xarray as xr
import datetime
import numpy as np
import math
import numpy.testing as npt

# this import attaches the extension methods to the DataArray class
import xarray_extensions.timeseries
import xarray_extensions.general

class Test(unittest.TestCase):

    def test_deseasonalised(self):
        # degenerate case with no seasonal variation, deseasonalisation should have no effect
        da1 = xr.DataArray(data=np.array([[[3 for i in range(1,13)],[4 for i in range(1,13)]]]),
                          dims=["lat","lon","time"],
                          coords={"time": [datetime.datetime(2003,i,1) for i in range(1,13)]})

        ds1 = da1.deseasonalised(clim=False, abs=True)
        npt.assert_almost_equal(da1.data, ds1.data)

        # perfectly seasonal pattern over 2 years
        da2 = xr.DataArray(data=np.array([[[1+math.sin(math.pi*2*i/12) for i in range(1, 25)], [3+math.cos(math.pi*2*i/12) for i in range(1, 25)]]]),
                           dims=["lat", "lon", "time"],
                           coords={"time": [datetime.datetime(2003 + (i-1)//12, 1+ ((i-1) % 12), 1) for i in range(1, 25)]})

        # check returned climatology
        ds2clim = da2.deseasonalised(clim=True)
        npt.assert_almost_equal(ds2clim.data, np.array([[[1+math.sin(math.pi*2*i/12) for i in range(1, 13)], [3+math.cos(math.pi*2*i/12) for i in range(1, 13)]]]))

        # should deseasonalise to 0
        ds2 = da2.deseasonalised(clim=False, abs=False)
        npt.assert_almost_equal(ds2.data, np.zeros((1,2,24)))

        # abs values should deseasonalise to constants 3, 4
        ds2abs = da2.deseasonalised(clim=False, abs=True)
        npt.assert_almost_equal(ds2abs.data, np.array([[[1.0 for i in range(1,25)],[3.0 for i in range(1,25)]]]))

    def test_detrended(self):
        # no trend, detrending should have no effect
        da1 = xr.DataArray(data=np.array([[[3 for i in range(1, 13)], [4 for i in range(1, 13)]]]),
                           dims=["lat", "lon", "time"],
                           coords={"time": [datetime.datetime(2003, i, 1) for i in range(1, 13)]})

        # check coeffs, constant term should be non-zero
        ds1coeffs = da1.detrended(coeff=True)
        exp_ds1coeffs = np.array([[[0, 3],[0, 4]]])
        npt.assert_almost_equal(ds1coeffs.data, exp_ds1coeffs)

        ds1 = da1.detrended(coeff=False, abs=True)
        npt.assert_almost_equal(da1.data, ds1.data)

        # generate a perfect linear trend, detrending should yield roughly constant (mean) values
        upward_trend = [0+(i-1) for i in range(1, 13)]
        downward_trend = [12-(i-1) for i in range(1, 13)]

        da2 = xr.DataArray(data=np.array([[upward_trend, downward_trend]]),
                           dims=["lat", "lon", "time"],
                           coords={"time": [datetime.datetime(2003, i, 1) for i in range(1, 13)]})

        exp_detrended = np.array([[[0.0 for i in range(1,13)],[0.0 for i in range(1, 13)]]])
        exp_detrended_abs = np.array(
            [[[np.mean(upward_trend) for i in range(1, 13)], [np.mean(downward_trend) for i in range(1, 13)]]])

        ds2 = da2.detrended(coeff=False, abs=False)
        ds2abs = da2.detrended(coeff=False, abs=True)
        # there is a bit of "noise" in the fitted coefficients
        npt.assert_almost_equal(exp_detrended, ds2.data, decimal=1)
        npt.assert_almost_equal(exp_detrended_abs, ds2abs.data, decimal=1)

        # for checking the coeffs, its easier to work with time as integers 0,1,2 etc
        da3 = xr.DataArray(data=np.array([[upward_trend, downward_trend]]),
                           dims=["lat", "lon", "time"],
                           coords={"time": [i for i in range(0, 12)]})

        ds3coeffs = da3.detrended(coeff=True, coeff_scale=None)
        exp_ds3coeffs = np.array([[[1,0],[-1,12]]])
        npt.assert_almost_equal(ds3coeffs.data, exp_ds3coeffs)

    def test_lagged_correlation(self):
        nlats = 5
        nlons = 8
        da = xr.DataArray(data=np.array(
            [[[math.sin(math.pi * 2 * i / 12) * (lon + 1) / (lat + 3) for i in range(1, 250)] for lon in range(nlons)]
             for lat in range(nlats)]),
                          dims=["lat", "lon", "time"],
                          coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                                           range(1, 250)],
                                  "lat": [lat for lat in range(nlats)],
                                  "lon": [lon for lon in range(nlons)]})
        da2 = da * 2

        da3 = da.lagged_correlation(da2,lags=[-6,-3,0,3,6])
        self.assertEqual(da3.dims, ('lat', 'lon', 'lag'))
        expected_correlations = np.array([[[-1,0,1,0,-1] for lon in range(nlons)] for lat in range(nlats)])
        npt.assert_almost_equal(da3.data,expected_correlations,decimal=3)
        npt.assert_equal(da3.coords["lat"],np.array([lat for lat in range(nlats)]))
        npt.assert_equal(da3.coords["lon"], np.array([lon for lon in range(nlons)]))

    def test_lagged_correlation_sig(self):
        nlats = 5
        nlons = 8
        da = xr.DataArray(data=np.array(
            [[[math.sin(math.pi * 2 * i / 12) * (lon + 1) / (lat + 3) for i in range(1, 250)] for lon in range(nlons)]
             for lat in range(nlats)]),
                          dims=["lat", "lon", "time"],
                          coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                                           range(1, 250)]})
        da2 = da * 2

        da3 = da.lagged_correlation(da2,lags=[-6,-3,0,3,6],ci=0.05,dof=100)
        self.assertEqual(da3.dims, ('lat', 'lon', 'lag', 'parameter'))
        expected_correlations = np.array([[[-1,0,1,0,-1] for lon in range(nlons)] for lat in range(nlats)])
        npt.assert_almost_equal(da3.isel(parameter=0).data,expected_correlations,decimal=3)

    def test_lagged_regression(self):
        nlats = 10
        nlons = 7
        da = xr.DataArray(data=np.array([[[math.sin(math.pi * 2 * i / 12)*(lon+1)/(lat+3) for i in range(1, 25)] for lon in range(nlons)] for lat in range(nlats)]),
                           dims=["lat", "lon", "time"],
                           coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                                            range(1, 25)],
                                   "lat": [lat for lat in range(nlats)],
                                   "lon": [lon for lon in range(nlons)]})
        da2 = da*2+1

        da3 = da.lagged_regression(da2,lags=[-6,-3,0,3,6])
        self.assertEqual(da3.dims,('lat','lon','lag','parameter'))

        npt.assert_almost_equal(da3.data[:, :, 0], np.array([[[-2,1] for lat in range(nlons)] for lon in range(nlats)]),decimal=1)
        npt.assert_almost_equal(da3.data[:, :, 2], np.array([[[2,1] for lat in range(nlons)] for lon in range(nlats)]), decimal=1)
        npt.assert_almost_equal(da3.data[:, :, 4], np.array([[[-2,1] for lat in range(nlons)] for lon in range(nlats)]), decimal=1)

        da4 = xr.DataArray(data=np.array([[[np.nan for i in range(1, 25)] for lon in range(nlons)] for lat in range(nlats)]),dims=["lat", "lon", "time"],
                           coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                                            range(1, 25)]})
        da5 = xr.DataArray(data=np.array([[[np.nan for i in range(1, 25)] for lon in range(nlons)] for lat in range(nlats)]),dims=["lat", "lon", "time"],
                           coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                                            range(1, 25)]})
        da6 = da4.lagged_regression(da5, lags=[-6, -3, 0, 3, 6])
        npt.assert_almost_equal(da6.data[:,:,0,:],
                                np.array([[[np.nan,np.nan] for lat in range(nlons)] for lon in range(nlats)]), decimal=1)
        npt.assert_equal(da3.coords["lat"],np.array([lat for lat in range(nlats)]))
        npt.assert_equal(da3.coords["lon"], np.array([lon for lon in range(nlons)]))


    def test_safe_assign(self):
        ds = xr.Dataset()
        ds["test1"] = xr.DataArray(data=np.array([1,2,3]),dims=["dim1"],coords={"dim1":[100,101,102]})
        ds["dim1"].attrs["metadata"] = "some metadata"
        ds.safe_assign(xr.DataArray(name="test2",data=np.array([3,4,5]),dims=["dim1"]))
        # following test would fail due to xarray defect https://github.com/pydata/xarray/issues/2245 if
        # we had used ds["test2"] = ... to assign the data array
        self.assertEqual(ds["dim1"].attrs["metadata"],"some metadata")

    def test_longitude_center_zero(self):
        ds = xr.Dataset()
        from random import random
        ds["test1"] = xr.DataArray(data=np.array([random() for i in range(0,360)]),dims=["lon"],coords={"lon":[i for i in range(0,360)]})
        ds["lon"].attrs["valid_min"] = 0
        ds["lon"].attrs["valid_max"] = 359
        ds2 = ds.longitude_center_zero()
        expected = ds["test1"].roll({"lon":-180}).data
        expected_lon = np.array([i for i in range(-180,180)])
        npt.assert_equal(expected,ds2["test1"].values)
        npt.assert_equal(expected_lon,ds2["lon"].values)
        self.assertEqual(ds2["lon"].attrs["valid_min"],-180)
        self.assertEqual(ds2["lon"].attrs["valid_max"],179)

if __name__ == '__main__':
    unittest.main()