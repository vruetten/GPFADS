import sys, time
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc import flatten
from mods.kernel import Kernel
from autograd.scipy.linalg import solve_triangular
from time import time
path = '/Users/virginiarutten/Documents/code/SCA/'
sys.path.append(path)
from scipy.optimize import minimize
from autograd import grad
from util.funcs import myqr
from models import mkernel3

class GPFA():
    def __init__(self, nlatent, kern = 'sq', yknt = None, xt = None):
        self.d = nlatent
        self.kern = kern
        self.fudge = 1e-6
        self.disp = True
        if yknt is not None:
            self.load_data(yknt, xt)

        self.initialse()

    def load_data(self, yknt, xt):
        self.yknt = yknt
        self.xt = xt
        self.k, self.n, self.t = yknt.shape
        assert self.t == xt.shape[0]

    def unpack_params(self, params, fudge = 1e-4):
        kern_params = self.kernel.unpack(params[0])
        C = params[1]
        C = myqr(C)[0] # if wanting orthogonal components
        R = params[2]
        mu = params[3]
        return kern_params, C, R, mu

    def initialse(self,):

        self.C = myqr(npr.randn(self.n, self.d))[0]
        self.r = np.ones([self.n])*.01
        self.mu = np.zeros([self.n])
        self.kernel = Kernel(d = self.d, kern = self.kern)
        

        ### set bounds ###
        Cb = [(-1e3, 1e3)]*self.d*self.n
        Rb = [(0.0001, 1e3)]*self.n
        mub = [(-1e3, 1e3)]*self.n

        self.params = [self.kernel.kern_params, self.C , self.r, self.mu]
    
        self.bounds = self.kernel.bounds + Cb + Rb + mub
        self.dict = {}
        self.dict['ll'] = []
        _, self.unflatten = flatten(self.params)


    def compute_iKy(self, params, yknt, xt):
        k, n, t = yknt.shape
        Idt = np.eye(self.d*t)
        It = np.eye(t)
        
        kern_params, C, r, mu = self.unpack_params(params)
        KXX = self.kernel.build_Kxx(xt, xt, kern_params)
        
        KXX = KXX + np.eye(KXX.shape[0])*self.fudge
        L = np.linalg.cholesky(KXX)
        iR12 = 1/(np.sqrt(r))
        A = L.T@(np.kron(C.T@np.diag(iR12), It)) # NT by NT

        B = Idt + L.T@(np.kron(C.T@np.diag(1/r)@C, It))@L
        M = np.linalg.cholesky(B)
        R_yk_nt = (yknt*iR12[None, :, None]).reshape([k, -1])
        Ay = (A@R_yk_nt.T).T  # k by ... 
        x = solve_triangular(M, Ay.T, lower = True)
        iBARy = solve_triangular(M.T, x, lower = False).T
        AiBARy = (A.T@iBARy.T).T
        Idt_AiBARy = R_yk_nt - AiBARy # k by nt
        x = (Idt_AiBARy.reshape([k, n, t])*iR12[None, :, None]).reshape([k, -1])
        return x, M, KXX


    def logevidence(self, params, yknt, xt):
        k, n, t = yknt.shape
        assert t == xt.shape[0]
        kern_params, C, r, mu = self.unpack_params(params)
        alphas = kern_params[1]

        yknt = yknt - mu[None,:,None]
        ykt = yknt.reshape([k, -1])
        
        iKy, M, Kxx  = self.compute_iKy(params, yknt, xt)
        
        yiKy = np.sum(ykt*iKy)
        logdet_ = np.sum(np.log(r))*t + 2*np.sum(np.log(np.diag(M))) 
        loge = -1/2*yiKy - 1/2*logdet_*k - k/2*np.log(2*np.pi)*t*n
        return loge/k 

    def objective(self, params):
        params = self.unflatten(params)
        return -self.logevidence(params, self.yknt, self.xt)

    def grad_obj(self, params):
       objective = lambda params : self.objective(params)
       mygrad = grad(objective)(params)
       return mygrad


    def optimise(self, params, num_ite = 1e5):
        t0 = time()
        ll0 = self.logevidence(params, self.yknt, self.xt)
        x0, _ = flatten(params)
        method = 'L-BFGS-B'
        # method = 'SLSQP'
        self.res = minimize(self.objective, x0 = x0, method = method, jac = self.grad_obj, options = {'disp':self.disp, 'maxiter': num_ite, 'gtol': 1e-7, 'ftol':1e-7}, callback = self.callback, bounds = self.bounds)

        self.params = self.unflatten(self.res['x'])
        self.params[1] = myqr(self.params[1])[0]
        self.dict['success'] = self.res['success']
        self.dict['nit'] = self.res['nit']         
        print('\nsuccess: {0}'.format(self.res['success']))
        print('\nnit: {0}'.format(self.res['nit']))
        print('\ncause: {0}'.format(self.res['message']))

        self.ll = self.logevidence(self.params, self.yknt, self.xt) 
        self.dict['params'] = self.params
        self.dict['ll'].append(self.ll)
        dur = (time()-t0)/60
        self.dict['dur'] = dur
        print('optimisation terminated... \nexecution time: {0}'.format(dur))


    def callback(self, xk):
        params = self.unflatten(xk)
        print(params[0][1][:2])
        ll = self.objective(xk)*-1
        self.dict['ll'].append(ll)
        pass

    def predx(self, yknt, params, xt):
        k, n, t = yknt.shape
        kern_params, C, R, mu = self.unpack_params(params)
        yknt = yknt - mu[None,:,None]
        iKy, M, _ = self.compute_iKy(params, yknt, xt)
        KXx = self.kernel.build_Kxx(self.xt, xt, kern_params)
        Kxx = self.kernel.build_Kxx(xt, xt, kern_params)
        iKyknt = iKy.reshape([k, n, t])
        Cv = C.T@iKyknt
        Cv = Cv.reshape([k, -1])
        tmp = np.kron(C, np.eye(t)).reshape([n, t, self.d*t]).transpose([2,0,1])
        iKC, _, _ = self.compute_iKy(params, tmp, xt)
        tmp = (C.T@iKC.reshape([-1, n, t])).reshape([-1, self.d*t])
        cov = Kxx - KXx.T@tmp@KXx
        dc = np.diag(cov)
        posterior_mean = (KXx.T@Cv.T).T
        t = xt.shape[0]
        return posterior_mean.reshape([k, -1, t]), (np.sqrt(dc + 1e-5)*2).reshape([self.d, -1])