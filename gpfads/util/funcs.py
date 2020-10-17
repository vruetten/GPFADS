import autograd.numpy as np
from autograd.scipy.special import dawsn
from myutils.grad_functions import imvoigt, myexpi, mylp_fdiff
from autograd.extend import primitive, defvjp


######## KERNEL FUNCTIONS  #########

### SPECTRAL MIXTURE ###
sm_fplus = lambda tau, l, w : np.exp(-1/2*(tau/l)**2)*np.cos(w*tau*2*np.pi)
sm_fdiff = lambda tau, l, w : np.exp(-1/2*(tau/l)**2)*np.sin(w*tau*2*np.pi) + np.exp(-1/2*(w*l*2*np.pi)**2)*imvoigt(tau/(l*np.sqrt(2)),(w*l*2*np.pi)/np.sqrt(2))

### EXPONENTIATED QUADRATIC ###
sq_fdiff = lambda tau, l, w : (2/np.sqrt(np.pi))*dawsn((tau/(l*np.sqrt(2))))
sq_fplus = lambda tau, l, w : np.exp(-1/2*(tau/l)**2)

### LAPLACIAN ###
lp_fdiff = lambda tau, l, w : mylp_fdiff(tau/l)
lp_fplus = lambda tau, l, w : np.exp(-np.abs(tau/l))

### COSINE ###
c_fdiff = lambda tau, l, w : np.sin(l*tau*2*np.pi)
c_fplus = lambda tau, l, w : np.cos(l*tau*2*np.pi)


ht_fdiff = lambda tau, l, w : tau*(1/(1 + tau**2))
ht_fplus = lambda tau, l, w : 1/(1 + tau**2)



########### SPECIAL FUNCTIONS AND THEIR GRADIENTS ###########

### IMVOIGT ### 
@primitive
def imvoigt(x, y):
    from scipy.special import wofz
    """imvoigt function"""
    return np.imag(wofz(x + 1j*y))

def imvoigt_vjp(g, ans, x, y):
    from scipy.special import wofz
    z = x + 1j*y
    zw = z*wofz(z)
    dx = -2*np.imag(zw) + 2/np.sqrt(np.pi)
    dy = -2*np.real(zw)
    return g*dx, g*dy

defvjp(imvoigt, \
    lambda ans, x, y: lambda g: imvoigt_vjp(g, ans, x, y)[0],\
    lambda ans, x, y: lambda g: imvoigt_vjp(g, ans, x, y)[1])


### EXPONENTIAL INTEGRAL ### 
@primitive
def myexpi(x):
    from scipy.special import expi
    """exponential integral function"""
    return expi(x)

def myexpi_vjp(g, ans, x):
    dx = np.exp(x)/x
    return g*dx

defvjp(myexpi, lambda ans, x: lambda g: myexpi_vjp(g, ans, x))

### FDIFF OF LAPLACIAN KERNEL ### 
@primitive
def mylp_fdiff(x):
    from scipy.special import expi
    fudge = 1e-10
    f2 = 700
    inds = np.abs(x)<fudge
    x[inds] = fudge
    x[x>f2] = f2
    x[x<-f2] = -f2
    tmp = 1/np.pi*(np.exp(-x)*expi(x) - np.exp(x)*expi(-x))
    tmp[inds] = 0
    return tmp

def mylp_fdiff_vjp(g, ans, x):
    from scipy.special import expi
    fudge = 1e-10
    f2 = 700
    inds = np.abs(x)<fudge
    x[inds] = fudge
    x[x>f2] = f2
    x[x<-f2] = -f2
    dx = -1/np.pi*(np.exp(x)*expi(-x) + np.exp(-x)*expi(x))
    dx[inds] = fudge*np.sign(x[inds])
    return g*dx

defvjp(mylp_fdiff, lambda ans, x: lambda g: mylp_fdiff_vjp(g, ans, x))


### QR ### 
@primitive
def myqr(x):
    from autograd.numpy import linalg as LA
    from autograd.numpy import copy
    """qr decomposition"""
    q,r = LA.qr(copy(x))
    return q, r


def myqr_vjp(g, ans, x):
    from autograd.numpy import matmul as m
    from autograd.numpy.linalg import inv
    gq = g[0]
    gr = g[1]
    q = ans[0]
    r = ans[1]

    rt = r.T
    rtinv = inv(rt)
    qt = q.T
    grt = gr.T
    gqt = gq.T

    mid = m(r,grt) - m(gr,rt)+ m(qt,gq)- m(gqt,q)

    n = mid.shape[0]
    indices = np.triu_indices(n, k = 0)
    tmp = np.ones([n,n])
    tmp[indices] = 0
    return m(q, gr + m(mid*tmp, rtinv)) + m((gq-m(q,m(qt,gq))),rtinv)

defvjp(myqr, lambda ans, x: lambda g: myqr_vjp(g, ans, x))
