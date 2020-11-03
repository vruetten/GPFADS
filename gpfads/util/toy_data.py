import numpy as np
import numpy.random as npr
from scipy.integrate import solve_ivp
from mods.kernel import Kernel
from util import util
from util import funcs
from scipy.stats import zscore

def draw_samples(d, k, t, tmax, kern, seed = 0):
    """
    
    """
    npr.seed(seed)
    xt = np.linspace(0, tmax, t)[:,None]
    kernel = Kernel(d = d, kern = kern)
    kern_params = kernel.kern_params
    kern_params[1] = kern_params[1]*0 + .99
    Kxx = kernel.build_Kxx(xt, xt, kern_params)
    yknt = util.sample(Kxx, k).reshape([k, d, -1])
    return yknt, xt

def sample_vdp(a = 0, b = 9, k = 100, t = 100):
    t = np.linspace(a, b, t)
    x = []
    for i in range(k):
        init = npr.randn(2)*1.5
        sol = solve_ivp(vdp, [a, b], [init[0], init[1]], t_eval=t)
        x.append(sol.y)
    x = np.array(x)
    d = {}
    d['xt'] = t[:,None]
    d['xkdt'] = x
    return d

def vdp(t, z):
    mu = 1
    x, y = z
    return [y, mu*(1 - x**2)*y - x]

def generate_vdp_data(n, sd_1, sd_2, tau1, tau2, t = 100, k = 100, seed = 0):
    npr.seed(seed)
    a = 0
    b = 12
    di = sample_vdp(a, b, k, t)
    k, d, t = di['xkdt'].shape
    latents = zscore(di['xkdt'].transpose([1,0,2]).\
        reshape([2,-1]), axis = -1).reshape([d, k, t]).transpose([1,0,2])
    xt = np.linspace(0, 1, t)[:,None]
    
    kern1 = np.exp(-.5*(xt-xt.T)**2/tau1**2)
    kern2 = np.exp(-.5*(xt-xt.T)**2/tau2**2)
    noise_k1 = npr.multivariate_normal(np.zeros([t]), kern1, k).reshape([k, 1, t])*sd_1
    noise_k2 = npr.multivariate_normal(np.zeros([t]), kern2, k).reshape([k, 1, t])*sd_1
    
    xknt = np.concatenate([latents, noise_k1, noise_k2], axis = 1)
    u = funcs.myqr(npr.randn(n, n))[0][:,:4]

    yknt = u.dot(xknt).transpose([1,0,2]) 
    yknt += npr.randn(*yknt.shape)*sd_2
    proj = np.concatenate([u[:,:2].T.dot(yknt),u[:,2:].T.dot(yknt)], axis = 0).transpose([1,0,2])
    return yknt, xknt, u, proj, xt
