import unittest
import xarray as xr
import datetime
import numpy as np
import math
import numpy.testing as npt

# this import attaches the extension methods to the DataArray class
import xarray_extensions.data_reduction

class Test(unittest.TestCase):

    def test_som(self):
        nlats = 400
        nlons = 400
        ntimes = 12
        data = np.random.randn(nlats,nlons,ntimes)
        da = xr.DataArray(data=data,dims=("lat","lon","time"),coords={"lat":range(0,nlats),"lon":range(0,nlons),"time":range(0,ntimes)})
        da2 = da.som(iters=1,verbose=True)
        self.assertEqual(da2.values.shape,(400,400,2))

    def test_pca(self):
        nlats = 400
        nlons = 400
        ntimes = 12
        data = np.random.randn(nlats, nlons, ntimes)
        da = xr.DataArray(data=data, dims=("lat", "lon", "time"),
                          coords={"lat": range(0, nlats), "lon": range(0, nlons), "time": range(0, ntimes)})
        da2 = da.pca(model_callback=lambda x: print(x.explained_variance_ratio_))
        self.assertEqual(da2.values.shape,(400,400,2))