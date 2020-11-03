import sys, time, os
path = '/Users/virginiarutten/Documents/code/GPFADS/gpfads/'
sys.path.append(path)
from util import toy_data as td
from util import util
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from mods.gpfa import GPFA

if __name__ == "__main__":
    figpath = '../plots/'
    rpath = '../data/result_vdp.npy'
    mysavefig = util.savefig(figpath = figpath)
    plot = False
    plot = True
    sample = True
    opt = True
    
    # sample = False
    # opt = False

    tau = [.09,0.05] # length-scales of Van Der Poll oscillator 
    sd_1 = 1 # latent variances
    sd_2 = .05 # observation noise

    n = 6 # embedding dimension
    k = 100 # number of trials
    t = 100
    seed = 0

    if sample:
        ### GENERATE AND SAVE TOY DATASET
        yknt, xknt, u, proj, xt = td.generate_vdp_data(n, sd_1, sd_2, *tau, t, k)
        k, n, t = yknt.shape
        k = k//2
        print(k, n, t)
        ytrain = yknt[:k]
        ytest = yknt[k:]
        k, n, t = ytrain.shape
        xt = np.linspace(0, 1, t)[:,None]
        data = {}
        data['yknt'] = yknt
        data['xknt'] = xknt
        data['xt'] = xt
        data['ytrain'] = ytrain
        data['ytest'] = ytest
        data['u'] = u
        np.save(rpath, data)
    else:
        data = np.load(rpath, allow_pickle = True).item()
        ytrain = data['ytrain']
        xt = data['xt']
        u = data['u']

    npr.seed(seed)
    nlatent = 4
    kern = 'sq'
    gp = GPFA(nlatent, kern = kern, yknt = ytrain, xt = xt)
    
    
    if opt:
        gp.optimise(gp.params)
        data['params'] = gp.params
        data['ll'] = gp.dict['ll']
        np.save(rpath, data)


    data = np.load(rpath, allow_pickle = True).item()
    ytrain = data['ytrain']
    ytest = data['ytest']
    xknt = data['xknt']
    params = data['params']
    xt = data['xt']
    u = data['u']
    xpred, _ = gp.predx(ytest, params, xt)

    print('\n length scales:')
    print(params[0][0])
    print('\n alphas:')
    print(params[0][1])
    print('\n sigmas:')
    print(params[0][2])
    print('\n')
    print(params[2])
    C_ = params[1]
    print(C_.T@u)


    if plot:
        x0 = xknt[:,:2]
        x1 = xknt[:,2:]
        nk = 5
        cmap = pl.cm.Set2(np.linspace(0, 1, nk))
        ###########################
        ## plot data
        ########################### 
        fig = pl.figure(figsize = (6,5))
        fig.suptitle('toy data', y = 1.05)
        gs = GridSpec(nrows = 2, ncols = 2) 
        ax0 = fig.add_subplot(gs[0,0])
        ax0.set_title('latent plane 1')
        ax0.set_xlabel('x1')
        ax0.set_ylabel('x2')
        ax1 = fig.add_subplot(gs[0,1], sharex = ax0)
        ax1.set_title('latent plane 2')
        ax1.set_xlabel('x3')
        ax1.set_ylabel('x4')

        ax2 = fig.add_subplot(gs[1,:]) 
        ax2.set_title('observations')
        
        [ax0.plot(x0[i,0], x0[i,1], color = cmap[i], lw = 2) for i in range(nk)]
        [ax1.plot(x1[i,0], x1[i,1], color = cmap[i], lw = 2) for i in range(nk)]
        [ax2.plot(xt, ytrain[0,i], color = cmap[i], lw = 2) for i in range(nk)]
        ax2.set_xlabel('time')
        pl.tight_layout()
        mysavefig('gpfa-data')

        ###########################
        ## log-lik
        ########################### 

        loglik = data['ll']
        cmap = pl.get_cmap('tab10')
        fig = pl.figure(figsize = (4, 2))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('optimisation \n cost function')
        ax.plot(loglik,'-o', c = cmap(0))
        ax.set_xlabel(r'ite \#')
        ax.set_ylabel('marginal\n loglikelihood')
        mysavefig('gpfa-loglik')

        ###########################
        ## plot inferred posterior means
        ###########################
        fig = pl.figure(figsize = (6, 5))
        fig.suptitle('inferred posterior mean on test data', y = 1.05)
        fs = 18
        nk = 10
        ax = pl.subplot(221)
        ax.set_title('latent plane 1')
        [ax.plot(xpred[i,0],xpred[i,1], ) for i in range(nk)]
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax = pl.subplot(222)
        ax.set_title('latent plane 2')
        [ax.plot(xpred[i,2],xpred[i,3]) for i in range(nk)]
        ax.set_xlabel('x3')
        ax.set_ylabel('x4')
        pl.tight_layout()
        mysavefig('gpfa-latents')
