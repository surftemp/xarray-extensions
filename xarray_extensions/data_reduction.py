"""
Defines an xarray extensions for data reduction.

Importing this module attaches extra methods to xarray DataArray as follows:

xarray.DataArray:
    som - perform PCA along a particular dimension and return the new map coordinates as a new DataArray
    pca - perform PCA along a particular dimension and return the components as a new DataArray
"""

import random
import sys
import xarray as xr
import logging
import numpy as np

sklearn_available = False
try:
    from sklearn.decomposition import PCA
    sklearn_available = True
except:
    logging.warning("scikit-learn is not installed - some methods will not be available")


class Progress(object):

    def __init__(self,label,silent=False):
        self.label = label
        self.last_progress_frac = None
        self.silent = silent

    def report(self,msg,progress_frac):
        if self.silent:
            return
        if self.last_progress_frac is None or (progress_frac - self.last_progress_frac) >= 0.01:
            self.last_progress_frac = progress_frac
            i = int(100*progress_frac)
            if i > 100:
                i = 100
            si = i // 2
            sys.stdout.write("\r%s %s %-05s %s" % (self.label,msg,str(i)+"%","#"*si))
            sys.stdout.flush()

    def complete(self,msg):
        if self.silent:
            return
        sys.stdout.write("\n%s %s\n" % (self.label,msg))
        sys.stdout.flush()


class SelfOrganisingMap(object):

    """
    Train Self Organising Map (SOM) with cells arranged in a hexagonal layout

    Parameters
    ----------
    iters : int
        the number of training iterations to use when training the SOM
    gridwidth : int
        number of cells across the grid
    gridheight : int
        number of cells down the grid
    initial_neighbourhood : int
        the initial neighbourhood size

    Keyword Parameters
    ------------------
    verbose : bool
        whether to print progress messages
    seed : int
        random seed - set to produce repeatable results
    """

    def __init__(self, gridwidth, gridheight, iters, initial_neighbourhood, verbose=False, seed=None):
        self.gridheight = gridheight
        self.gridwidth = gridwidth
        self.iters = iters
        self.initial_neighbourhood = initial_neighbourhood
        self.verbose = verbose
        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)
        self.learn_rate_initial = 0.5
        self.learn_rate_final = 0.05

    def get_weights(self,outputIndex):
        return self.weights[:,outputIndex].tolist()

    def fit_transform(self, instances):
        self.neighbour_limit = 0
        self.nr_inputs = instances.shape[1]
        self.nr_instances = instances.shape[0]
        self.instance_mask = ~np.any(np.isnan(instances), axis=1)

        self.nr_outputs = self.gridwidth * self.gridheight
        self.nr_weights = self.nr_outputs * self.nr_inputs

        self.weights = np.zeros((self.nr_inputs, self.nr_outputs))
        for row in range(0, self.nr_inputs):
            for col in range(0, self.nr_outputs):
                self.weights[row, col] = self.rng.random()

        p = Progress("SOM",silent=not self.verbose)
        progress_frac = 0.0
        p.report("Starting", progress_frac)
        iteration = 0
        while iteration < self.iters:
            learn_rate = (1.0 - float(iteration) / float(self.iters)) \
                         * (self.learn_rate_initial - self.learn_rate_final) + self.learn_rate_final
            neighbour_limit = self.initial_neighbourhood - int(
                (float(iteration) / float((self.iters + 1))) * self.initial_neighbourhood)
            logging.debug("iter=%d (of %d) / learning-rate=%f / neighbourhood=%d"%(iteration, self.iters,
                                                                                   learn_rate,
                                                                                   neighbour_limit))
            for i in range(self.nr_instances):
                if self.instance_mask[i]:
                    winner = self.compute_activations(instances[i, :])
                    self.update_network(winner, instances[i, :], neighbour_limit, learn_rate)

            iteration += 1
            progress_frac = iteration/self.iters
            p.report("Training neighbourhood=%d"%(neighbour_limit), progress_frac)

        p.complete("SOM Training Complete")

        scores = np.zeros(shape=(self.nr_instances, 2))

        for i in range(self.nr_instances):
            if self.instance_mask[i]:
                winner = self.coords(self.compute_activations(instances[i, :]))
            else:
                winner = [np.nan,np.nan]
            scores[i,:] = np.array(winner)

        return scores

    def compute_activations(self,values):
        inarr = np.expand_dims(values, axis=1)
        sqdiffs = (self.weights - inarr) ** 2
        sumsdiffs = np.sum(sqdiffs, axis=0)
        return np.argmin(sumsdiffs)

    def update_network(self, winner, values, neighbour_limit, learn_rate):
        (wx,wy) = self.coords(winner)
        for x in range(max(0,wx-neighbour_limit),min(self.gridwidth, wx+neighbour_limit+1)):
            for y in range(max(0, wy - neighbour_limit), min(self.gridheight, wy + neighbour_limit + 1)):
                index = self.get_output(x, y)
                self.weights[:,index] -= learn_rate * (self.weights[:, index]-values)

    def coords(self, output):
        return (output % self.gridwidth, output // self.gridwidth)

    def get_output(self, x, y):
        return x + (y*self.gridwidth)


def som(self, dimension="time", gridwidth=16, gridheight=16, iters=25, initial_neighbourhood=3, seed=1,
        verbose=False, model_callback=None):
    """
    Extract and return the self organising map coordinates for this data array after reducing along a target dimension

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound
    dimension: str
       the name of the target dimension to be reduced
    iters : int
        the number of training iterations to use when training the SOM
    gridwidth : int
        number of cells across the grid
    gridheight : int
        number of cells down the grid
    initial_neighbourhood : int
        the initial neighbourhood size
    seed : int
        random seed - set to produce repeatable results
    verbose : bool
        whether to print progress messages (SOM can take some time to run)
    model_callback: function
       if provided, this method will be invoked with a SelfOrganisingMap object
       (see source code of this module)

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance with the target dimension replaced by a dimension of size 2

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    stack_dims = tuple(x for x in self.dims if x != dimension)
    stack_sizes = tuple(self.values.shape[i] for i in range(len(self.dims)) if self.dims[i] != dimension)
    instances = self.stack(case=stack_dims).transpose("case",dimension)
    s = SelfOrganisingMap(gridwidth,gridheight,iters,initial_neighbourhood,seed=seed,verbose=verbose)
    scores = s.fit_transform(instances.values)
    a = scores.reshape(stack_sizes+(2,))
    new_coords = {}
    for key in self.coords:
        if key != dimension:
            new_coords[key] = self.coords[key]
    new_dims = stack_dims+("som_axis",)
    if model_callback:
        model_callback(s)
    return xr.DataArray(data=a, dims=new_dims, coords=new_coords)


def pca(self, dimension="time", n_components=2, model_callback=None):
    """
    Extract and return the principle components for this data array after reducing along a target dimension

    Parameters
    ----------
    self: xarray.DataArray
       the DataArray instance to which this method is bound
    dimension: str
       the name of the target dimension to be reduced
    n_components: int
       the number of components to extract, which must be less than the length of the target dimension
    model_callback: function
       if provided, this method will be invoked with a sklearn.decomposition.PCA object after fitting
       (see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

    Returns
    -------
    xarray.DataArray
       an xarray.DataArray instance with the target dimension replaced by a dimension of a size
       controlled by the n_components parameter

    Notes
    -----

    This function is attached to the DataArray class as a method when this module is imported
    """
    stack_dims = tuple(x for x in self.dims if x != dimension)
    stack_sizes = tuple(self.values.shape[i] for i in range(len(self.dims)) if self.dims[i] != dimension)

    instances = self.stack(case=stack_dims).transpose("case", dimension).values
    (nr_instances, nr_values) = instances.shape

    # for any instances containing at least one NaNs, record those and replace all values with dummy (zero) values.
    # PCA cannot handle NaNs.
    mask = np.all(~np.isnan(instances),axis=1)
    instance_mask = np.expand_dims(mask,1)
    instance_mask = np.tile(instance_mask,nr_values)
    instances = np.where(instance_mask,instances,0)

    pca = PCA(n_components=n_components)
    factor_scores = pca.fit_transform(instances)

    # for instances that had NaNs replaced with zero, replace the dummy factor scores with NaN.
    factor_scores_mask = np.expand_dims(mask, 1)
    factor_scores_mask = np.tile(factor_scores_mask, n_components)
    factor_scores = np.where(factor_scores_mask, factor_scores, np.nan)

    # reshape the factor scores back to the original axes
    a = factor_scores.reshape(stack_sizes + (n_components,))

    # create a DataArray object for output
    new_coords = {}
    for key in self.coords:
        if key != dimension:
            new_coords[key] = self.coords[key]
    new_dims = stack_dims+("pca_component",)
    if model_callback:
        model_callback(pca)
    return xr.DataArray(data=a, dims=new_dims, coords=new_coords)


from xarray_extensions.utils import bind_method

bind_method(xr.DataArray, som)
if sklearn_available:
    bind_method(xr.DataArray, pca)

