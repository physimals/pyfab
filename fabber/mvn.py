import math

from fsl.data.image import Image

class MVN(Image):

    def __init__(self, *args, **kwargs):
        Image.__init__(self, *args, **kwargs)
        self.nparams = self._get_num_params()

    def _get_num_params(self):
        if self.ndim != 4:
            raise ValueError("MVN images must be 4D")
        nz = float(self.shape[3])
        nparams = (math.sqrt(1+8*nz) - 1) / 2 - 1

        if nparams < 1 or nparams != float(int(nparams)):
            raise ValueError("Invalid number of volumes for MVN image: %i" % nz)
        return nparams

    def mean_index(self, param_idx):
        """
        Return index of mean value of a parameter

        :param param_idx: Parameter index, first parameter is zero
        :return Volume index containing mean values of the parameter (first volume is zero)
        """
        if param_idx >= self.nparams:
            raise ValueError("Invalid parameter index: %i (number of parameters: %i)" % (param_idx, self.nparams))
        return self.nparams*(self.nparams+1)/2 + param_idx

    def var_index(self, param_idx, cov_param_idx=None):
        """
        Return index of variance/covariance value of a parameter

        :param param_idx: Parameter index, first parameter is zero
        :param cov_param_idx: If covariance required, index of parameter to find covariance with respect to. 
                              If not specified return index of variance (i.e. take cov_param_idx = param_idx).
        :return Volume index containing specified variance/covariance value of the parameter (first volume is zero)
        """
        if cov_param_idx is None:
            cov_param_idx = param_idx
        
        row = max(param_idx, cov_param_idx)
        col = min(param_idx, cov_param_idx)
        start_idx = row * (row+1) / 2
        return start_idx + col
