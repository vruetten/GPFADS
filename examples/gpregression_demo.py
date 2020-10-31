import os
import sys, time
path = '/Users/virginiarutten/Documents/code/GPFADS/gpfads/'
sys.path.append(path)
from models.gpregression import GPregression
import numpy as np
import numpy.random as npr
from util import toy_data as td
from util import util
from autograd.misc import flatten
import matplotlib.pyplot as pl
import matplotlib as mpl
from copy import deepcopy
from math import nan


if __name__ == "__main__":
    figpath = '../plots/'
    rpath = './result.npy'
    mysavefig = util.savefig(figpath = figpath)

    k = 25
    ktest = 1
    d = 2 
    t = 100     
    tmax = 5
    kern = 'cos'
    plot = True
    
    seed0, seed1 = 0, 1
    yknt, xt = td.draw_samples(d, k, t, tmax, kern, seed = seed0)
    yknt_test, xt = td.draw_samples(d, ktest, t, tmax, kern, seed = seed1)

    xtest = np.linspace(0, tmax, t)[:,None]
    sd = .1
    yknt += npr.randn(*yknt.shape)*sd
    yknt_test += npr.randn(*yknt_test.shape)*sd
    reg = GPregression(yknt, xt, mean = False, kern = kern)

    # reg.optimise(reg.params)
    # np.save(rpath, reg.dict)    
    res = np.load(rpath, allow_pickle = True).item()
    params = res['params']

    yknt_test_cond = deepcopy(yknt_test)
    yknt_test_cond[:,1,20:] = nan # select which variables to hold out
    mu, cov = reg.pred(xt, xt, yknt_test_cond, params)
    sdp = mu + cov[None]*2
    sdm = mu - cov[None]*2


    if plot:

        pl.rcParams["axes.prop_cycle"] = pl.cycler("color", pl.cm.Set2(np.linspace(0,1,7)))
        cmap = pl.get_cmap('Set2')
        fig = pl.figure(figsize = (6,2))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('extrapolations', y = 1.1)
        ax.plot(xtest, yknt_test[:,1].T,'--', label = 'ground truth - x2', ms = 2,  c = 'k', alpha = .3)
        ax.plot(xtest, yknt_test_cond[:,0].T,'-o', label = 'observations - x1', ms = 2, c = cmap(0))
        ax.plot(xtest, mu[:,1].T, label = 'posterior mean', c = cmap(1))
        ax.plot(xtest, yknt_test_cond[:,1].T,'o', label = 'observations - x2', ms = 2,  c = cmap(1))

        ax.fill_between(xtest[:,0], sdp[0,1], sdm[0,1], color = cmap(2), alpha = .4, label = '2 std')
        
        ax.legend(fontsize = 8,bbox_to_anchor=(1.2, 1.6), loc='upper right', ncol=1)
        ax.set_aspect(.3)
        mysavefig('reg-extrapolations')

        loglik = res['ll']
        cmap = pl.get_cmap('tab10')
        fig = pl.figure(figsize = (4, 2))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('optimisation \n cost function')
        ax.plot(loglik,'-o', c = cmap(0))
        ax.set_xlabel(r'ite \#')
        ax.set_ylabel('marginal\n loglikelihood')
        
        mysavefig('reg-loglik')


        fig = pl.figure(figsize = (2,2))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('data samples')
        nk = 3
        [ax.plot(yknt[i,0].T, yknt[i,1].T) for i in range(nk)]
        ax.axis('equal')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_aspect(1)
        mysavefig('reg-data')

