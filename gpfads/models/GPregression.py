import autograd.numpy as np
from autograd import grad
from autograd.misc import flatten
from scipy.optimize import minimize
from autograd.numpy.linalg import inv, slogdet, svd, eigh, cholesky, eig
import autograd.scipy.stats.multivariate_normal as mvn
from time import time
import matplotlib.pyplot as pl
from util import util
from models.kernel import Kernel
import sys

class GPregression():
    def __init__(self, ykdt, xt, mean = False, kern = 'sq'):
        self.ykdt = ykdt
        self.k, self.d, self.t = ykdt.shape
        self.xt = xt
        self.kern = kern
        self.mean = mean
        self.fudge = 1e-4
        self.disp = True
        self.initialise()

    def initialise(self):

        self.kernel = Kernel(d = self.d, kern = self.kern)
        self.wnoise = np.ones([self.d])*0.5
        self.mean_params = np.ones([3, self.d])*1e-3
        wnoiseb = [(1e-2, 1e2)]*self.d
        meanpb = [(-1e2, 1e2)]*3*self.d

        self.params = [self.kernel.kern_params, self.wnoise, self.mean_params]
        self.bounds = self.kernel.bounds + wnoiseb + meanpb

        self.ykt_ = self.ykdt.reshape([self.k, -1])

        self.dict = {}
        self.dict['ll'] = []
        _, self.unflatten = flatten(self.params)

    def build_Kxx(self, x0, x1, params, prior = False):
        kern_params, wnoise, mean_params = self.unpack_params(params)
        Kxx = self.kernel.build_Kxx(x0, x1, kern_params)
        if prior:
            Kxx += np.kron(np.diag(wnoise), np.eye(x0.shape[0]))
        return Kxx

    def logevidence(self, params):
        kern_params, wnoise, mean_params = self.unpack_params(params, fudge = self.fudge)
        Kxx = self.build_Kxx(self.xt, self.xt, params, prior = True)

        L = cholesky(Kxx)
        iL = inv(L)
        inv_Kxx = iL.T@iL
        if self.mean:
            mu = self.mean(self.xt, params)[None] # D x T
            yc = (self.ykdt - mu).reshape([self.k, -1])
        else:
            yc = self.ykt_
        
        logdet = self.k*np.sum(np.log(np.diag(L)))*2
        ll = -1/2*np.sum((yc@inv_Kxx)*yc) - 1/2*logdet - self.k/2*np.log(2*np.pi)*self.t*self.d
        # lp = mvn.logpdf(yc, yc[0].squeeze()*0, Kxx).sum() # check marg-log-likelihood
        return ll.sum()

    def unpack_params(self, params, fudge = 1e-4):
        kern_params = self.kernel.unpack(params[0])
        wnoise = params[1].squeeze()
        mean_params = params[2]
        return kern_params, wnoise, mean_params

    def objective(self, params):
        params = self.unflatten(params)
        return -self.logevidence(params)

    def grad_obj(self, params):
       objective = lambda params : self.objective(params)
       return grad(objective)(params)

    def callback(self, xk):
        params = self.unflatten(xk)
        ll = self.objective(xk)*-1
        self.dict['ll'].append(ll)
        # self.dict['params'].append(params)
        # kern_params, wn, mn = self.unpack_params(params)
        # print(wn)
        pass

    def optimise(self, params, num_ite = 1e4):
        t0 = time()
        ll0 = self.logevidence(params)
        x0, _ = flatten(params)
        self.dict['ll'].append(ll0)
        
        self.res = minimize(self.objective, x0 = x0, method = 'L-BFGS-B', jac = self.grad_obj, options = {'disp':self.disp, 'maxiter': num_ite, 'gtol': 1e-7, 'ftol':1e-8}, callback = self.callback, bounds = self.bounds)

        self.params = self.unflatten(self.res['x'])
        self.dict['success'] = self.res['success']
        self.dict['nit'] = self.res['nit']         
        print('\nsuccess: {0}'.format(self.res['success']))
        print('\nnit: {0}'.format(self.res['nit']))
        print('\ncause: {0}'.format(self.res['message']))

        self.ll = self.logevidence(self.params) 
        self.dict['params'] = self.params
        self.dict['ll'].append(self.ll)
        dur = (time()-t0)/60
        self.dict['dur'] = dur
        print('optimisation terminated... \nexecution time: {0}'.format(dur))

    def pred(self, xt, x1, ykdt, params):
        
        kern_params, wnoise, mean_params = self.unpack_params(params, fudge = self.fudge)
        k, d, t = ykdt.shape
        if self.mean:
            mu = self.mean(self.xt, params)[None] # D x T ### TO BE CHANGED
            yc = (ykdt - mu).reshape([self.k, -1])
        else:
            yc = ykdt.reshape([k, -1])

        
        KXX = self.build_Kxx(xt, xt, params, prior = True) 

        # select points to condition on
        val_inds = np.argwhere(np.isnan(yc[0]) == False).squeeze()
        nval_inds = np.argwhere(np.isnan(yc[0]) == True).squeeze()
        KXX = KXX[:,val_inds]
        KXX = KXX[val_inds]
        yc = yc[:,val_inds]
        
        L = np.linalg.cholesky(KXX)
        iL = inv(L)
        Kinv = iL.T@iL 

        KXx = self.build_Kxx(xt, x1, params, prior = False)
        t0 = xt.shape[0]
        t1 = x1.shape[0]
        KXx = KXx.reshape([t0, d, t1, d]).reshape([t0*d, -1])
        noise = np.kron(np.diag(wnoise), np.eye(t0))
        noise[nval_inds, nval_inds] = 0
        KXx[:t0*d,:t0*d] += noise
        KXx = KXx.reshape([t0, d, t1, d]).reshape([d*t0, -1])
        KXx = KXx[val_inds]

        Kxx = self.build_Kxx(x1, x1, params, prior = True)
        mu_pred = KXx.T.dot(Kinv).dot(yc.T).T
        cov_pred = Kxx - KXx.T.dot(Kinv).dot(KXx)
        mu_pred = mu_pred.reshape([k, d, -1])
        return mu_pred, np.sqrt(np.diag(cov_pred).reshape([d, -1]) + self.fudge)