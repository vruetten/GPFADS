import numpy as np
from gpfads.models.kernel import Kernel
from gpfads.util import util

def draw_samples(d, k, t, tmax, kern):
    """
    docstring
    """
    xt = np.linspace(0, tmax, t)[:,None]
    kernel = Kernel(d = d, kern = kern)
    kern_params = np.copy(kernel.kern_params)
    Kxx = kernel.build_Kxx(xt, xt, kern_params)
    yknt = util.sample(Kxx, k).reshape([k, d, -1])
    return yknt, xt

