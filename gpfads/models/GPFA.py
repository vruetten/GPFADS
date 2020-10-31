import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc import flatten


class GPFA():
    def __init__(self, nlatent, kernel = 'sq', yknt = None, xt = None):
        self.d = nlatent
        self.kernel = kernel
        if yknt is not None:
            self.load_data(yknt, xt)


    def load_data(self, yknt, xt):
        self.yknt = yknt
        self.xt = xt
        k, n, t = yknt.shape
        assert t == xt.shape[0]

    def initialse(self,):
        self.C = npr.randn(self.d, self.n)
        self.r = np.ones([self.d])*.001
        self.mu = np.zeros([self.d])

        ### set bounds ###
        Cb = [(-1e3, 1e3)]*self.d*self.n
        Rb = [(0.0001, 1e3)]*self.d
        mub = [(-1e3, 1e3)]*self.d
        

        self.params = [self.kern.kern_params, self.C , self.r, self.mu]
        x0, self.unflatten = flatten(self.params)

        self.bounds = self.kern.bounds + Rb + Cb + mu


    def compute_iKy(self, yknt, xt, params):
        k, n, t = yknt.shape
        Idt = np.eye(self.d*t)
        It = np.eye(t)
        
        kern_params, C, r, mu = self.unpack_params(params)
        KXX = self.kern.build_Kxx(xt, xt, kern_params)
        
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
        
        iKy, M, Kxx  = self.compute_iKy(yknt, params, xt)
        
        yiKy = np.sum(ykt*iKy)
        logdet_ = np.sum(np.log(r))*t + 2*np.sum(np.log(np.diag(M))) 
        loge = -1/2*yiKy - 1/2*logdet_*k - k/2*np.log(2*np.pi)*t*n
        return loge/k 

    def objective(self, params):
        params = self.unflatten(params)
        return -self.logevidence(params)

    def grad_obj(self, params):
       objective = lambda params : self.objective(params)
       mygrad = grad(objective)(params)
       return mygrad