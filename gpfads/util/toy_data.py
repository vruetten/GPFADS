import numpy as np
import numpy.random as npr
from models.kernel import Kernel
from util import util

def draw_samples(d, k, t, tmax, kern, seed = 0):
    """
    docstring
    """
    npr.seed(seed)
    xt = np.linspace(0, tmax, t)[:,None]
    kernel = Kernel(d = d, kern = kern)
    kern_params = kernel.kern_params
    kern_params[1] = kern_params[1]*0 + .99
    Kxx = kernel.build_Kxx(xt, xt, kern_params)
    yknt = util.sample(Kxx, k).reshape([k, d, -1])
    return yknt, xt

