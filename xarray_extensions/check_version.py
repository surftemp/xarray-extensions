import xarray as xr
checked = False

def check_version():
    # some functionality is broken on xarray < 0.19.0,
    # so raise an exception if an earlier version of xarray is installed
    global checked
    if not checked:
        maj_min_rel = xr.__version__.split(".")
        (maj,min,rel) = tuple(map(lambda x:int(x),maj_min_rel))
        if maj == 0 and min < 19:
            raise Exception("[xarray-extensions] WARNING, xarray >= 0.19.0 is required but current version is %s" % xr.__version__)
        checked = True

