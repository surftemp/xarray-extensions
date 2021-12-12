if __name__ == '__main__':
    nlats = 400
    nlons = 400
    ntimes = 2
    data = np.random.randn(nlats, nlons, ntimes)
    da = xr.DataArray(data=data, dims=("lat", "lon", "time"),
                      coords={"lat": range(0, nlats), "lon": range(0, nlons), "time": range(0, ntimes)})
    da.rgb_plot(dimension="time")