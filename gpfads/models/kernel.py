import numpy as np
from gpfads.util import funcs

def intialise_kernel_params(d, kern = 'sq', kk = 1):
    ''' d: number of outputs
        kk: # of length-scale per plane
        '''
    alpha = np.ones((d, 1))*0
    alphab =[(0, .999)]*d
    l = np.ones((d, kk))*1
    lb = [(1e-2, 1e3)]*d*kk

    if 'sm' in kern:
        w = np.ones((d, kk))*1
        wb = [(1e-3, 10)]*d*kk

        bounds = lb + alphab + wb
        return [l, alpha, w], bounds

    else:
        bounds = lb + alphab 
        return [l, alpha], bounds 

def build_A(d):
    '''
    builds skew symmetric matrix with upper diag full of 1s, lower diag full of -1, and diag = 0
    '''
    A = np.tri(d, M = d, k = -1).T # strict upper triangle of 1s
    A -= A.T # strict lower triangle of 1s
    return A

def build_E(d):
    nd = d//2
    Es = np.zeros([np.max([1,nd]), d, d])
    ii = 0
    for i in np.arange(0,d,2):
        j = i+1
        inds = np.array([[i,i],[j,j],[i,j],[j,i]])
        Es[ii,inds[:,0],inds[:,1]] = 1
        ii += 1
    return Es


class Kernel():
    def __init__(self, d, k = 1, kern = 'sq', fudge = 1e4):
        self.d = d
        self.k = k
        self.kern = kern
        self.fudge = fudge

        self.initialise()

    
    def initialise(self):
        self.kern_params, self.bounds = intialise_kernel_params(d = self.d, kern = self.kern, kk = self.k)

        self.Id = np.eye(self.d)
        self.A = build_A(self.d)
        self.Es = build_E(self.d)

        if 'sq' in self.kern:
            self.fdiff = funcs.sq_fdiff
            self.fplus = funcs.sq_fplus
            print('using sq-exp kernel')
        if 'cos' in self.kern:
            self.fdiff = funcs.c_fdiff
            self.fplus = funcs.c_fplus
        if 'lp' in self.kern:
            self.fdiff = funcs.lp_fdiff
            self.fplus = funcs.lp_fplus
            print('using laplacian kernel')
        if 'sm' in self.kern:
            self.fdiff = funcs.sm_fdiff
            self.fplus = funcs.sm_fplus
            print('using spectral mixture kernel')

    def unpack(self, params, ell_floor = 0.001, constrain = False):
        return params

    def build_plane(self, ind, kern_params, x0, x1):
        '''
        ind: plane number
        '''
        if 'sm' in self.kern:
            l, alpha, w = self.unpack(kern_params)
            pp = list(zip(l[ind], w[ind]))
        else:
            l, alpha = self.unpack(kern_params)
            pp = list(zip(l[ind]))
        
        
        fplus, fdiff = self.build_F(x0, x1, pp, self.fplus, self.fdiff, split = False)

        Aplus = self.Id*self.Es[ind] # selects appropriate diagonal elements
        Adiff = self.A*self.Es[ind] # selects appropriate off-diagonal elements

        cplus = np.kron(Aplus, fplus)
        cdiff = alpha[ind]*np.kron(Adiff, fdiff)
        kxx = cplus + cdiff
        return kxx

    def build_Kxx(self, x0, x1, kern_params, fudge = 1e-4, split = False):
        Kxx = []
        for ind in range(self.d//2):
            kxx = self.build_plane(ind, kern_params, x0, x1)
            Kxx.append(kxx)
        return np.sum(Kxx, axis = 0)


    def build_F(self, x0, x1, params, fplus, fdiff, split = False):
        tau = x0 - x1.T
        t = x0.shape[0]
        fplus =  np.array([fplus(tau, *p) for p in params])
        fdiff =  np.array([fdiff(tau, *p) for p in params])
        if split:
            return fplus, fdiff
        else:
            return np.sum(fplus, axis = 0), np.sum(fdiff, axis = 0)
