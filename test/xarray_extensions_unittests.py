import random
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

    def test_lagged_correlation_1D(self):
        dts = [datetime.datetime(2013,4,15,12,0,0),
               datetime.datetime(2013,5,15,12,0,0),
               datetime.datetime(2013,6,15,12,0,0),
               datetime.datetime(2013,7,15,12,0,0),
               datetime.datetime(2013,8,15,12,0,0)]
        lead_values = [1,2,0,-4,-5]
        lag_values =  [7,1,2,0,-4]
        lead_da = xr.DataArray(data=np.array([[lead_values]]),dims=["lat", "lon", "time"],
                               coords={"time": dts,"lat": [0],"lon": [0]})
        lag_da = xr.DataArray(data=np.array([[lag_values]]), dims=["lat", "lon", "time"],
                               coords={"time": dts, "lat": [0], "lon": [0]})
        da3 = lag_da.lagged_correlation(lead_da,lags=[1])
        expected_correlations = np.array([[[1]]])
        npt.assert_almost_equal(da3.data,expected_correlations,decimal=3)
        da4 = lead_da.lagged_correlation(lag_da, lags=[-1])
        npt.assert_almost_equal(da4.data, expected_correlations, decimal=3)

    def test_lagged_correlation(self):
        nlats = 5
        nlons = 8
        da = xr.DataArray(data=np.array(
            [[[math.sin(math.pi * 2 * i / 12) * (lon + 1) / (lat + 3) for i in range(1, 250)] for lon in
              range(nlons)]
             for lat in range(nlats)]),
            dims=["lat", "lon", "time"],
            coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                             range(1, 250)],
                    "lat": [lat for lat in range(nlats)],
                    "lon": [lon for lon in range(nlons)]})
        da2 = da * 2
        da3 = da.lagged_correlation(da2, lags=[-6, -3, 0, 3, 6])
        self.assertEqual(da3.dims, ('lat', 'lon', 'lag'))
        expected_correlations = np.array([[[-1, 0, 1, 0, -1] for lon in range(nlons)] for lat in range(nlats)])
        npt.assert_almost_equal(da3.data, expected_correlations, decimal=3)
        npt.assert_equal(da3.coords["lat"], np.array([lat for lat in range(nlats)]))
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

    def test_timeaxis_check(self):
        # calls to lagged_correlation|regression[_month_of_year] should check that the participating datasets
        # are aligned exactly on their time axis

        # prepare misaligned da1 and da2 and check that each of the methods throw the correct exception
        # also check that invalid values for month_of_year also cause the correct exception to be thrown
        nlats = 10
        nlons = 7
        da1 = xr.DataArray(data=np.array(
            [[[0 for i in range(1, 25)] for lon in range(nlons)] for lat in range(nlats)]),
                          dims=["lat", "lon", "time"],
                          coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                                           range(1, 25)],
                                  "lat": [lat for lat in range(nlats)],
                                  "lon": [lon for lon in range(nlons)]})
        da2 = xr.DataArray(data=np.array([[[0 for i in range(1, 25)] for lon in range(nlons)] for lat in range(nlats)]),
            dims=["lat", "lon", "time"],
            coords={"time": [datetime.datetime(2004 + (i - 1) // 12, 1 + ((i - 1) % 12), 1) for i in
                             range(1, 25)],
                    "lat": [lat for lat in range(nlats)],
                    "lon": [lon for lon in range(nlons)]})

        try:
            da1.lagged_correlation(da2,lags=[1])
            self.assertTrue(False,"time coordinates check has failed")
        except Exception as ex:
            self.assertTrue(isinstance(ex,xarray_extensions.timeseries.MisalignedTimeAxisException))

        try:
            da1.lagged_regression(da2,lags=[1])
            self.assertTrue(False,"time coordinates check has failed")
        except Exception as ex:
            self.assertTrue(isinstance(ex,xarray_extensions.timeseries.MisalignedTimeAxisException))

        try:
            da1.lagged_correlation_month_of_year(da2,lags=[1],month_of_year=6)
            self.assertTrue(False,"time coordinates check has failed")
        except Exception as ex:
            self.assertTrue(isinstance(ex,xarray_extensions.timeseries.MisalignedTimeAxisException))

        try:
            da1.lagged_regression_month_of_year(da2,lags=[1],month_of_year=6)
            self.assertTrue(False,"time coordinates check has failed")
        except Exception as ex:
            self.assertTrue(isinstance(ex,xarray_extensions.timeseries.MisalignedTimeAxisException))

        try:
            da1.lagged_regression_month_of_year(da1,lags=[1],month_of_year=0)
            self.assertTrue(False,"month_of_year check has failed")
        except Exception as ex:
            self.assertTrue(isinstance(ex,xarray_extensions.timeseries.InvalidMonthOfYearException))

        try:
            da1.lagged_regression_month_of_year(da1,lags=[1],month_of_year=13)
            self.assertTrue(False,"month_of_year check has failed")
        except Exception as ex:
            self.assertTrue(isinstance(ex,xarray_extensions.timeseries.InvalidMonthOfYearException))


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

        da3 = da2.lagged_regression(da,lags=[-6,-3,0,3,6])
        self.assertEqual(da3.dims,('lat','lon','lag','parameter'))

        npt.assert_almost_equal(da3.data[:, :, 0], np.array([[[-2,1] for lat in range(nlons)] for lon in range(nlats)]),decimal=3)
        npt.assert_almost_equal(da3.data[:, :, 2], np.array([[[2,1] for lat in range(nlons)] for lon in range(nlats)]), decimal=3)
        npt.assert_almost_equal(da3.data[:, :, 4], np.array([[[-2,1] for lat in range(nlons)] for lon in range(nlats)]), decimal=3)

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

    def test_lagged_correlation_month_of_year_timeseries(self):
        # test that passing in a 1D array produces the same results as passing in the same array, expanded to 3D
        nlats = 50
        nlons = 80
        ntimes = 48
        rng = random.Random(0)
        da = xr.DataArray(data=np.array(
            [[[rng.random() for i in range(1, ntimes)] for lon in range(nlons)]
             for lat in range(nlats)]),
                          dims=["lat", "lon", "time"],
                          coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 15,12,0,0) for i in
                                           range(1, ntimes)],
                                  "lat": [lat for lat in range(nlats)],
                                  "lon": [lon for lon in range(nlons)]})

        da2 = xr.DataArray(data=np.array([rng.random() for i in range(1, ntimes)]), dims=["time"],
                           coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 15, 12, 0, 0)
                                            for i in range(1, ntimes)]})

        da3 = da2.expand_dims({"lat":nlats,"lon":nlons},axis=[0,1])

        for moy in range(1,13):
            da4 = da.lagged_correlation_month_of_year(da2,lags=[-1,0,1],month_of_year=moy)
            self.assertEqual(da4.dims, ('lat', 'lon', 'lag'))
            da5 = da.lagged_correlation_month_of_year(da3, lags=[-1, 0, 1], month_of_year=moy)
            self.assertEqual(da5.dims, ('lat', 'lon', 'lag'))
            npt.assert_equal(da4.data,da5.data)

    def test_lagged_correlation_month_of_year(self):
        nlats = 50
        nlons = 80
        ntimes = 48
        rng = random.Random(0)
        da = xr.DataArray(data=np.array(
            [[[rng.random() for i in range(1, ntimes)] for lon in range(nlons)]
             for lat in range(nlats)]),
                          dims=["lat", "lon", "time"],
                          coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 15,12,0,0) for i in
                                           range(1, ntimes)],
                                  "lat": [lat for lat in range(nlats)],
                                  "lon": [lon for lon in range(nlons)]})

        da2 = da.shift({"time": -1}) # da2 is da, shifted 1 month "later" (da2 lags da by 1 month)
        da2neg = da.shift({"time": 12}) * -1 # da2neg is da2 shifted 12 months "later" and inverted (da2neg leads da by 12 months)

        for moy in range(1,13):
            da3 = da.lagged_correlation_month_of_year(da,lags=[-1,0,1],month_of_year=moy)
            self.assertEqual(da3.dims, ('lat', 'lon', 'lag'))
            expected_correlations = np.array([[1 for lon in range(nlons)] for lat in range(nlats)])
            npt.assert_almost_equal(da3.sel(lag=0).data,expected_correlations,decimal=3)

            try:
                npt.assert_almost_equal(da3.sel(lag=-1).data, expected_correlations, decimal=3)
                self.assertTrue(False, "Array contents should NOT be almost equal")
            except AssertionError:
                pass

            try:
                npt.assert_almost_equal(da3.sel(lag=1).data, expected_correlations, decimal=3)
                self.assertTrue(False, "Array contents should NOT be almost equal")
            except AssertionError:
                pass

            npt.assert_equal(da3.coords["lat"],np.array([lat for lat in range(nlats)]))
            npt.assert_equal(da3.coords["lon"], np.array([lon for lon in range(nlons)]))

            da4 = da.lagged_correlation_month_of_year(da2, lags=[1], month_of_year=1)
            expected_correlations = np.array([[[1] for lon in range(nlons)] for lat in range(nlats)])
            npt.assert_almost_equal(da4.data, expected_correlations, decimal=3)

            da5 = da.lagged_correlation_month_of_year(da2neg, lags=[-12], month_of_year=moy)
            expected_correlations = np.array([[[-1] for lon in range(nlons)] for lat in range(nlats)])
            npt.assert_almost_equal(da5.data, expected_correlations, decimal=3)

            da6 = da.lagged_correlation_month_of_year(da2neg, lags=[11], month_of_year=moy)
            expected_avg_correlations = 0
            npt.assert_almost_equal(da6.data.mean(), expected_avg_correlations, decimal=1)

    def test_lagged_regression_month_of_year(self):
        nlats = 50
        nlons = 80
        ntimes = 48
        rng = random.Random(0)
        da = xr.DataArray(data=np.array(
            [[[rng.random() for i in range(1, ntimes)] for lon in range(nlons)]
             for lat in range(nlats)]),
                          dims=["lat", "lon", "time"],
                          coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 15, 12, 0 ,0) for i in
                                           range(1, ntimes)],
                                  "lat": [lat for lat in range(nlats)],
                                  "lon": [lon for lon in range(nlons)]})

        da2 = da.shift({"time": -1}) *2 # da2 is da, shifted 1 month "later" (da2 lags da by 1 month)
        da2neg = da.shift(
            {"time": 12}) * -1 + 3  # da2neg is da2 shifted 12 months "later" and inverted (da2neg leads da by 12 months)

        for moy in range(1,13):
            da3 = da.lagged_regression_month_of_year(da,lags=[-1,0,1],month_of_year=moy)
            self.assertEqual(da3.dims, ('lat', 'lon', 'lag', 'parameter'))
            expected_coefficients = np.array([[[1,0] for lon in range(nlons)] for lat in range(nlats)])
            npt.assert_almost_equal(da3.sel(lag=0).data,expected_coefficients,decimal=3)

            try:
                npt.assert_almost_equal(da3.sel(lag=-1).data, expected_coefficients, decimal=3)
                self.assertTrue(False,"Array contents should NOT be almost equal")
            except AssertionError:
                pass

            try:
                npt.assert_almost_equal(da3.sel(lag=1).data, expected_coefficients, decimal=3)
                self.assertTrue(False,"Array contents should NOT be almost equal")
            except AssertionError:
                pass

            npt.assert_equal(da3.coords["lat"], np.array([lat for lat in range(nlats)]))
            npt.assert_equal(da3.coords["lon"], np.array([lon for lon in range(nlons)]))

            da4 = da2.lagged_regression_month_of_year(da, lags=[-1], month_of_year=moy)
            expected_coefficients = np.array([[[[2,0]] for lon in range(nlons)] for lat in range(nlats)])
            npt.assert_almost_equal(da4.data, expected_coefficients, decimal=3)

            da5 = da2neg.lagged_regression_month_of_year(da, lags=[12], month_of_year=moy)
            expected_coefficients = np.array([[[[-1,3]] for lon in range(nlons)] for lat in range(nlats)])
            npt.assert_almost_equal(da5.data, expected_coefficients, decimal=3)

    def test_lagged_regression_month_of_year_timeseries(self):
        # test that passing in a 1D array produces the same results as passing in the same array, expanded to 3D
        nlats = 50
        nlons = 80
        ntimes = 48
        rng = random.Random(0)
        da = xr.DataArray(data=np.array(
            [[[rng.random() for i in range(1, ntimes)] for lon in range(nlons)]
             for lat in range(nlats)]),
                          dims=["lat", "lon", "time"],
                          coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 15,12,0,0) for i in
                                           range(1, ntimes)],
                                  "lat": [lat for lat in range(nlats)],
                                  "lon": [lon for lon in range(nlons)]})

        da2 = xr.DataArray(data=np.array([rng.random() for i in range(1, ntimes)]), dims=["time"],
                           coords={"time": [datetime.datetime(2003 + (i - 1) // 12, 1 + ((i - 1) % 12), 15, 12, 0, 0)
                                            for i in range(1, ntimes)]})

        da3 = da2.expand_dims({"lat":nlats,"lon":nlons},axis=[0,1])
        da3 = da3.assign_coords({"lat": [lat for lat in range(nlats)],
                                  "lon": [lon for lon in range(nlons)]})

        for moy in range(1,13):
            da4 = da.lagged_regression_month_of_year(da2,lags=[-1,0,1],month_of_year=moy)
            self.assertEqual(da4.dims, ('lat', 'lon', 'lag', 'parameter'))
            da5 = da.lagged_regression_month_of_year(da3, lags=[-1, 0, 1], month_of_year=moy)
            self.assertEqual(da5.dims, ('lat', 'lon', 'lag', 'parameter'))
            npt.assert_equal(da4.data,da5.data)

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