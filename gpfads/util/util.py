import autograd.numpy as np
import autograd.numpy.random as npr
from toolz import curry
import os


def sample(kern, nsamples):
    samples = npr.multivariate_normal(np.zeros([kern.shape[0]]), kern, nsamples)
    return samples

def is_pd(K):
    from autograd.numpy.linalg import cholesky
    try:
        cholesky(K)
        print('Matrix IS positive definite')
        return 1 
    except:
        print('Matrix is NOT positive definite')
        return 0
    
@curry
def savefig(fname, figpath):
    import matplotlib.pyplot as pl
    ext = '.png'
    fpath = figpath + fname + ext
    pl.savefig(fpath)
    command = 'open ' + fpath
    # os.system(command)