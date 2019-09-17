"""
Multivariate normal distribution processing
===========================================

This module provides support for manipulation of a voxelwise multivariate normal 
distribution (MVN) as returned by Fabber as its posterior output (finalMVN)

The intention of this module is to replace the functionality of the
mvntool program, i.e. to support editing parameter mean and variance
values.
"""

import math

import numpy as np

try:
    from fsl.data.image import Image
except ImportError:
    # Without fslpy this module will not work, however we don't
    # want to break imports so this will do as a kludge...
    Image = object

class MVN(Image):
    """
    FSL Image subclass representing a voxelwise multivariate normal distribution

    An MVN is a 4D image whose 3D component corresponds to the voxels in the parameter
    maps and whose 4D volumes encode the parameter means, variances and covariances.
    The latter are stored as the upper triangle of a (nparams+1)x(nparams+1) matrix 
    formed by concatenating the covariances with the last column/row containing the 
    means. The bottom right element is 1 by convention.

    For this reason the number of volumes in an MVN is given by 0.5*(nparams+1)*(nparams+2)
    and the number of parameters represented in an MVN is derived by inverting this
    formula. 

    Note that the parameters consist of the model parameters followed by noise parameters.
    Generally there is only one noise parameter so the number of model parameters
    is ``nparams-1``. However this is not guaranteed since some noise models can have
    multiple noise parameters.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor. 

        Can take same arguments as the fsl.data.image.Image constructor
        but also supports keyword ``param_names`` argument. This should
        be a list of parameter names which matches the size of the MVN
        """
        self.param_names = kwargs.get("param_names", None)
        Image.__init__(self, *args, **kwargs)
        self.nparams = self._get_num_params()
        if self.param_names is None:
            self.param_names = ["Parameter %i" % idx for idx in range(self.nparams)]
        elif len(self.param_names) != self.nparams:
            raise ValueError("MVN contains %i parameters, but %i parameter names were provided")

    def _get_num_params(self):
        if self.ndim != 4:
            raise ValueError("MVN images must be 4D")
        nz = float(self.shape[3])
        nparams = int((math.sqrt(1+8*nz) - 1) / 2 - 1)

        if nparams < 1 or nparams != float(int(nparams)):
            raise ValueError("Invalid number of volumes for MVN image: %i" % nz)
        return nparams

    def _get_param_idx(self, param):
        if isinstance(param, int):
            return param
        elif param in self.param_names:
            return self.param_names.index(param)
        else:
            raise ValueError("Parameter not found: %s - not an index or a recognized parameter name" % str(param))

    def _as_np_array(self, val):
        try:
            # Convert constant value into a Numpy array of the same 3D shape as the MVN
            val = float(val)
            return np.full(self.shape[:3], val)
        except (ValueError, TypeError):
            # Mean can't be converted to float so assume it's a Numpy array already
            return val

    def mean_index(self, param):
        """
        Return index of mean value of a parameter

        :param param: Parameter - either an index (first parameter is zero) or parameter name
        :return Volume index containing mean values of the parameter (first volume is zero)
        """
        param_idx = self._get_param_idx(param)

        if param_idx >= self.nparams:
            raise ValueError("Invalid parameter index: %i (number of parameters: %i)" % (param_idx, self.nparams))
        return self.nparams*(self.nparams+1)/2 + param_idx

    def var_index(self, param, cov_param=None):
        """
        Return index of variance/covariance value of a parameter

        :param param: Parameter - either an index (first parameter is zero) or parameter name
        :param cov_param: If covariance required, parameter to find covariance with respect to. 
                          (index or parameter name). If not specified return index of variance 
                          (i.e. take cov_param = param).
        :return Volume index containing specified variance/covariance value of the parameter (first volume is zero)
        """
        if cov_param is None:
            cov_param = param

        param_idx = self._get_param_idx(param)
        cov_param_idx = self._get_param_idx(cov_param)
        
        row = max(param_idx, cov_param_idx)
        col = min(param_idx, cov_param_idx)
        start_idx = row * (row+1) / 2
        return start_idx + col

    def update(self, param, mean=None, var=None):
        """
        Update the mean and/or variance for a parameter

        :param param: Parameter - either an index (first parameter is zero) or parameter name
        :param mean: If specified either a single value or a correctly sized Numpy array containing
                     the new mean value for this parameter
        :param var: If specified either a single value or a correctly sized Numpy array containing
                    the new variance for this parameter
        """
        if mean is not None:
            self[..., self.mean_index(param)] = self._as_np_array(mean)
        if var is not None:
            self[..., self.var_index(param)] = self._as_np_array(var)
