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
