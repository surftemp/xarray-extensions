
xarray 0.19.0 or later is required, attempting to import these extensions with an earlier version of xarray will cause
an exception to the thrown.  These methods extend xarray with timeseries and various other functionality.

.. currentmodule:: xarray_extensions.timeseries

==============================
DataArray methods - timeseries
==============================

include `import xarray_extensions.timeseries` in your code to add these methods to xarray.DataArray

.. autofunction:: deseasonalised
.. autofunction:: detrended
.. autofunction:: lagged_correlation
.. autofunction:: lagged_correlation_month_of_year
.. autofunction:: lagged_regression
.. autofunction:: lagged_regression_month_of_year

.. currentmodule:: xarray_extensions.data_reduction

==================================
DataArray methods - data reduction
==================================

include `import xarray_extensions.data_reduction` in your code to add these methods to xarray.DataArray

.. autofunction:: pca
.. autofunction:: som

.. currentmodule:: xarray_extensions.plots

=========================
DataArray methods - plots
=========================

include `import xarray_extensions.plots` in your code to add these methods to xarray.DataArray

.. autofunction:: rgb_plot

.. currentmodule:: xarray_extensions.general

=========================
Dataset Methods - general
=========================

include `import xarray_extensions.general` in your code to add these methods to xarray.Dataset

.. autofunction:: safe_assign
.. autofunction:: longitude_center_zero
