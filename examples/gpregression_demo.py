from gpfads.models.gpregression import GPregression
import autograd.numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from gpfads.util import toy_data as td
from gpfads.util import util


if __name__ == "__main__":
    figpath = './plots/'
    mysavefig = util.savefig(figpath = figpath)

    k = 2 
    d = 2 
    t = 100     
    tmax = 5
    kern = 'cos'
    yknt, xt = td.draw_samples(d, k, t, tmax, kern)


    reg = GPregression(yknt, xt, mean = False, kern = 'cos')

    params_init = reg.params
    l0 = reg.logevidence(params_init)
    print(l0)

    reg.optimise(params_init)
    l1 = reg.logevidence(reg.params)
    print(l1)


    # fig = pl.figure(figsize = (2,2))
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(yknt[0].T, yknt[1].T)
    # ax.axis('equal')
    # ax.set_aspect(1)
    # mysavefig('test')
