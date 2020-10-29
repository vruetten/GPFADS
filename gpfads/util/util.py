import numpy as np
import numpy.random as npr


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