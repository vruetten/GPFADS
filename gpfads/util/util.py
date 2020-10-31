import numpy as np
import numpy.random as npr
from toolz import curry
import os


def sample(kern, nsamples):
    samples = npr.multivariate_normal(np.zeros([kern.shape[0]]), kern, nsamples)
    return samples

def is_pd(K):
    from numpy.linalg import cholesky
    from numpy import linalg
    try:
        cholesky(K)
        return 1 
    except:
        print('Matrix is not positive definite')
        return 0
    
@curry
def savefig(fname, figpath):
    import matplotlib.pyplot as pl
    ext = '.png'
    fpath = figpath + fname + ext
    pl.savefig(fpath)
    command = 'open ' + fpath
    os.system(command)