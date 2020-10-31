import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as pl
import matplotlib as mpl
from gpfads.models.kernel import Kernel
from gpfads.util import util
import os

if __name__ == "__main__":
    figpath = './plots/'
    mysavefig = util.savefig(figpath = figpath)

    '''
    builds reversible and non-reversible covariance matrix (Kxx)
    and draws k samples from each and plots
    
    k: # of samples
    d: # of outputs
    t: # of timepoints
    '''

    k = 10 
    d = 2 
    t = 100 
    tmax = 5
    kern = 'sq'
    alpha = .99
    plot = True

    xt = np.linspace(0, tmax, t)[:,None]
    kernel = Kernel(d = d, kern = kern)

    kern_rev_params = np.copy(kernel.kern_params)
    kern_nonrev_params = np.copy(kernel.kern_params)
    kern_nonrev_params[1] = kern_nonrev_params[1]*0 + alpha 

    Kxx_rev = kernel.build_Kxx(xt, xt, kern_rev_params)
    Kxx_nonrev = kernel.build_Kxx(xt, xt, kern_nonrev_params)

    yknt_rev = util.sample(Kxx_rev, k).reshape([k, d, -1])
    yknt_nonrev = util.sample(Kxx_nonrev, k).reshape([k, d, -1])


    if plot:    
        ############ PLOT RESULTS ############
        pl.rcParams["axes.prop_cycle"] = pl.cycler("color", pl.cm.Set2(np.linspace(0,1,k)))

        fname = 'cov.png'
        fig, axs = pl.subplots(ncols = 2, figsize = (4, 2))
        fig.suptitle('Multi-ouptut covariance', y = 1.2)
        axs[0].imshow(Kxx_rev)
        axs[1].imshow(Kxx_nonrev)
        axs[0].set_title('reversible \n' + r'$\alpha = 0$')
        axs[1].set_title('non-reversible \n'+ r'$\alpha = $' + str(alpha))
        [axs[i].axis('off') for i in range(2)]
        mysavefig(fname)


        fname = 'draws.png'
        fig, axs = pl.subplots(ncols = 2, figsize = (4, 2), sharex = True, sharey = True)
        fig.suptitle('Samples', y = 1.3)
        axs[0].plot(yknt_rev[:,0].T, yknt_rev[:,1].T)
        axs[1].plot(yknt_nonrev[:,0].T, yknt_nonrev[:,1].T)
        axs[0].set_title('reversible \n' + r'$\alpha = 0$')
        axs[1].set_title('non-reversible \n'+ r'$\alpha = $' + str(alpha))
        mysavefig(fname)




