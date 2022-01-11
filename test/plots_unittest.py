import unittest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import random
import math

import xarray_extensions.plots
import xarray_extensions.data_reduction

class Test(unittest.TestCase):

    def generate_test_data(self,nlats,nlons,ntimes,nclusters=10,seed=1):
        rng = random.Random(seed)
        cluster_vectors = np.zeros(shape=(nclusters,ntimes))
        cluster_centers = np.zeros(shape=(nclusters,2),dtype=int)
        for n in range(nclusters):
            theta = rng.random()*2*math.pi
            for t in range(ntimes):
                v = math.sin(theta + t*0.25)
                cluster_vectors[n,t] = v
            cluster_centers[n,0] = int(rng.random()*nlats)
            cluster_centers[n,1] = int(rng.random()*nlons)

        data = np.random.randn(nlats, nlons, ntimes)
        clusters = list(range(nclusters))
        for lat in range(nlats):
            for lon in range(nlons):
                cluster_sqdistance = None
                cluster_nearest = None
                for cluster in clusters:
                    lat_diff = lat - cluster_centers[cluster,0]
                    lon_diff = lon - cluster_centers[cluster,1]
                    sq_dist = lat_diff*lat_diff + lon_diff*lon_diff
                    if cluster_nearest is None or sq_dist < cluster_sqdistance:
                        cluster_nearest = cluster
                        cluster_sqdistance = sq_dist
                data[lat,lon,:] = cluster_vectors[cluster_nearest,:]

        da = xr.DataArray(data=data, dims=("lat", "lon", "time"),
            coords={"lat": range(0, nlats), "lon": range(0, nlons), "time": range(0, ntimes)})
        return da


    def test_rgb(self):

        nlats = 400
        nlons = 400
        ntimes = 2

        da = self.generate_test_data(nlats,nlons,ntimes,nclusters=10)
        da.rgb_plot(dimension="time")
        plt.savefig("rgb.png")
        # plt.show()

    def test_pca_rgb(self):
        nlats = 400
        nlons = 400
        ntimes = 12
        da = self.generate_test_data(nlats,nlons,ntimes,nclusters=10)
        da2 = da.pca(model_callback=lambda x: print(x.explained_variance_ratio_))
        da2.rgb_plot(dimension="pca_component")
        plt.savefig("pca_rgb.png")

if __name__ == '__main__':
    unittest.main()