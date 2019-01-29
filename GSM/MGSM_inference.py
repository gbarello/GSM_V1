import time
import math
import numpy as np
import scipy
#import mpmath as mp
from scipy.integrate import quad as INT
import scipy.optimize
import scipy.linalg 
import theano.tensor as T
import theano

def mpbesselk(v,x):
    return scipy.special.kv(v,np.clip(x,0,700))#= np.frompyfunc(mp.besselk,2,1)

def mplog(x):
    return np.log(x)#= np.frompyfunc(mp.log,1,1)

import integrate as logint

def dot(x):
    if len(x) == 1:
        return x
    elif len(x) == 2:
        return np.dot(x[0],x[1])
    else:
        return np.dot(x[0],dot(x[1:]))

'''

this module contains functions needed for learning the MGSM and doing inference on the MGSM

'''

INF_TYPES = ["clean","noisy","AR_clean","AR_noisy"]

AHIGH = 1000
ALOW = .00001
NPT = 100000


LIMIT = 10000.

eps = .01
EPS = .001

def sigmoid(x):
      return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
      return np.exp(-x) / np.power(1. + np.exp(-x),2)

def integrate_log(f,low,high,npnt,logsp = False):
    '''
    return the log of teh integral of exp(f)
    '''
    if logsp:
        A = np.logspace(np.log10(low),np.log10(high),npnt)
    else:
        A = np.linspace(low,high,npnt)
    val = np.array([[a,f(a)] for a in A])
    fa = (val[:-1,1] + val[1:,1])/2
    aa = val[1:,0]-val[:-1,0]
    V = np.log(aa) + fa
    mm = np.max(V)
        
    return np.log(np.sum(np.exp(V - mm))) + mm

def expectation_log(E,f,pnts,logsp = False,n_d_exp = 1):
    '''
    return the expectation of E under Exp[f]
    '''
    
    A = pnts
    
    val = np.array([[a,f(a)] for a in A])
    Eval = np.array([E(a) for a in A])
    fa = (val[:-1,1] + val[1:,1])/2
    Ea = (Eval[:-1] + Eval[1:])/2
    aa = val[1:,0]-val[:-1,0]
    V = np.log(aa) + fa
    mm = np.max(V)


    norm = np.sum(np.exp(V - mm))

    temp = np.exp(V - mm)
    
    for k in range(n_d_exp):
        temp = np.expand_dims(temp,-1)
    
    inte = np.sum(Ea*temp,axis = 0)

    out = inte/norm
    return out

def find_f_max(f,low=0,high=np.inf,start=1,eps = 1e-5):
    dx = eps
    mid = start
    new_mid = mid+dx

    
    fi = f(mid)
    fm = f(new_mid)

    step = 0
    while np.abs(dx) > 1e-10 and step < 1000 and np.abs(mid - low) > 1e-7 and np.abs(mid-high)>1e-7 and np.abs(fi - fm) > 1e-10:

        if fm < fi:
            dx *= -.5
        else:
            dx *=1.1

        mid = new_mid
            
        step+=1

        new_mid = mid + dx

        
        if new_mid <= low:
            new_mid = (mid + low)/2
        elif new_mid >= high:
            new_mid = (mid + high)/2
            
        fi = f(mid)
        fm = f(new_mid)
        
    d2 = get_2nd_deriv(f,mid)

    return mid,f(mid),d2[0],d2[1],step

def get_2nd_deriv(f,pnt,einit = .1):

    if pnt - einit <= 0:
        eps = pnt/10
    else:
        eps = einit
    dd = [f(pnt-eps),f(pnt),f(pnt+eps)]
    d2 = (dd[2] - 2*dd[1] + dd[0])/(eps*eps)
    d1 = (dd[2] - dd[0])/(2*eps)
    '''
    step = 0

    while step < 100 and eps > 1e-5:
        eps = eps/2
        dd = [f(pnt-eps),f(pnt),f(pnt+eps)]

        d2_new = (dd[2] - 2*dd[1] + dd[0])/(eps*eps)

        if np.abs(d2_new - d2) < 1e-5:
            break
        d2 = d2_new
        print(d2,eps)
    '''
    return d2,d1

def rectify(x):
    return (x + np.abs(x))/2

def check_seg(seg):
    a = [np.concatenate(s) for s in seg]

    ll = len(a[0])

    for l in a:
        if len(l) != ll:
            return 1

    ll = range(ll)

    for l in a:
        for c in l:
            if c not in ll:
                return 2

    return 0

###

def att_PIA_iter(I,a,cov,ncov,qcov,F,log = False):
    n = len(cov)
    m = len(I)

    iF = np.linalg.inv(F)

    s,logdetF = np.linalg.slogdet(F)

    em = I[0]/a
    xi = ncov/(a*a)
    
    out = - n*m*np.log(a)

    for k in range(m-1):
        tempcov = (ncov/a/a) + np.dot(np.dot(np.transpose(F),xi + qcov),F)
        temparg = np.dot(iF,em) - (I[k+1]/a)
        
        s,logdet = np.linalg.slogdet(tempcov)
        
        out -= n*np.log(2*np.pi)/2 + logdetF
        out -= logdet/2
        out -= np.dot(np.dot(temparg,np.linalg.inv(tempcov)),temparg)/2
        
        em,xi = get_new_mc(em,xi,I[k+1]/a,ncov/a/a,qcov,F)

    s,logdet = np.linalg.slogdet(xi + cov)
    
    out -= n*np.log(2*np.pi)/2
    out -= logdet/2
    out -= np.dot(np.dot(em,np.linalg.inv(xi + cov)),em)/2

    if log:
        return(out)
    else:
        return np.exp(out)


def inv(x):
    return np.linalg.inv(x)

def get_new_mc(m,x,mu,cov,qcov,F):
    tempcov = np.dot(np.dot(np.transpose(F),x+qcov),F)
    xi = inv(inv(cov) + inv(tempcov))
    em = np.dot(xi,np.dot(inv(cov),mu)+np.dot(inv(tempcov),np.dot(inv(F),m)))

    return em,xi

def F_self_con(cov,qcov):
    return np.linalg.cholesky(np.identity(len(cov)) - np.dot(np.linalg.inv(cov),qcov))

def Q_self_con(cov,F):
    return cov - np.dot(np.transpose(F),np.dot(cov,F))

'''
coef = -5.
a,b,d,c=find_f_max(lambda x:(coef/2)*(x - 100)**2)
print(a,b,d,c)
print(coef)

print(np.exp(b)*(np.sqrt(2*np.pi)*np.sqrt(-1./d)))
print(INT(lambda x:np.exp((coef/2)*(x - 100)**2),0,200))
exit()
'''
##main inference functions

def GSM_gexp(I,data,inf_type):

    assert inf_type in INF_TYPES
    assert "cov" in data.keys()

    if inf_type == "clean":
        return gnn(I,data["cov"])
    elif inf_type == "noisy":
        assert "ncov" in data.keys()
        return gexp(I,data["cov"],data["ncov"])
    elif inf_type == "AR_noisy":
        assert "ncov" in data.keys()
        assert "qcov" in data.keys()
        return att_gexp(I,data["cov"],data["ncov"],data["qcov"])
    else:
        print("GSM_gexp inference type failed to materialize.")
        exit()

def MGSM_gexp(I,data,inf_type):

    assert inf_type in INF_TYPES
    assert "cov" in data.keys()
    assert "prob" in data.keys()

    if inf_type == "clean":
        return MGSM_gnn(I,data["cov"],data["prob"])
    elif inf_type == "noisy":
        assert "ncov" in data.keys()
        return MGSM_g(I,data["cov"],data["ncov"],data["prob"])
    elif inf_type == "AR_noisy":
        assert "ncov" in data.keys()
        assert "qcov" in data.keys()
        return MGSM_att_g(I,data["cov"],data["ncov"],data["qcov"],data["prob"])
    else:
        print("GSM_gexp inference type failed to materialize.")
        exit()

##helper functions

def weighted_f_cov(F,weights = None):
    '''
    Description: This function takes a set of filters for an MGSM model and possibly a set of weights and computes the full covariance
    '''
    
    S = F.shape

    return np.cov(np.transpose(np.reshape(F,(S[0],np.product(S[1:])))),aweights = weights)

def weighted_cov(F,weights = []):
    S = F.shape

    if len(weights) > 0:        
        fp = np.array([F[f] for f in range(len(F)) if np.isfinite(weights[f])])
        wp = np.array([weights[f] for f in range(len(F)) if np.isfinite(weights[f])])
        return np.cov(np.transpose(fp),aweights = wp)
    else:
        fp = F
        wp = weights
        return np.cov(np.transpose(fp))


def IP(x,C,y,keepdims = True):
    
    if len(x.shape) == 2:
        return (x*np.transpose(np.tensordot(C,y,axes=[1,1]))).sum(axis = 1,keepdims = keepdims)

    if len(x.shape) == 1:
        return (x*np.tensordot(C,y,axes=[1,0])).sum(axis = 0)

def get_matrix_data(name):
    F = open(name,"r")
    out = []

    for l in F:
        out.append([])

        temp = l[:-1].split(',')
        for x in temp:
            out[-1].append(float(x))

    F.close()
    return np.array(out)
    
def get_array_data(name):
    F = open(name,"r")
    out = []

    for l in F:
        out.append([])

        temp = l[2:-4].split('}","{')
#        print(temp)
#        exit()
        #temp ~ [n_site]
        for t in range(len(temp)):
            temp[t] = temp[t].split(",")
        #temp ~ [n_site,nfilt]
        for x in temp:
            #x ~ [n_filt]
            out[-1].append([float(t) for t in x])

    F.close()
    return np.array(out)

##helper functions specific to this MGSM implementation

def LAM(X,C):
    return np.sqrt(IP(X,C,X))

def H_V(v,cov):
    return (np.identity(cov.shape[0]) + (1./v**2)*np.linalg.inv(cov))

def AI(x1,x2,cov):
    return gnn(x1 + x2,cov)/(gnn(x1,cov) + gnn(x2,cov))
 
##helper functions that manipulate filters

def rot_filt(x,i):
    #for this implementation I am going to have the filters arranged as follows:
    #[n_samples,n_sites,n_ori,n_phase]

    shape = x.shape
    #print(shape)

    temp = x[:,[0] + np.roll(np.arange(1,shape[1]),i).tolist()]
    return np.roll(temp,i * (2 * shape[2])/shape[1],2)

def get_seg_x(X,n):

    shape = X.shape

    if n == 0:
        return np.reshape(X,[-1,np.prod(shape[1:])]),[]
    else:
        temp = rot_filt(X,n - 1)

        A = np.reshape(temp[:,:6],[-1,6*np.prod(shape[2:])])
        B = np.reshape(temp[:,6:],[-1,(shape[1] - 6)*np.prod(shape[2:])])
            
        return A,B

def get_CC_seg_x(xin,n):

    #I need to return the coen-cagli segmentations
    #there are 5
    #0 - center segmented together, surrounding orientations separately (5 segments)
    #1-4 - center and a single surrround angle segmented together. (4 segments) 
    #
    #

    X = np.reshape(xin,[-1,9,4,2])
    shape = X.shape

    if n == 0:
        #center segmented separately from the rest

        return [np.reshape(X[:,0],[shape[0],-1])] + [np.reshape(X[:,1:,i],[shape[0],-1]) for i in range(shape[2])]

    else:
        #now the center is segmented along with a surround angle, and the other angles are separate.
        return [np.concatenate([np.reshape(X[:,0],[shape[0],-1]),np.reshape(X[:,1:,n-1],[shape[0],-1])],axis = 1)] + [np.reshape(X[:,1:,i],[shape[0],-1]) if i != n - 1 else [] for i in range(shape[2])]

def PIA(n,x,a,cov,ncov,prec = None ,ret_exp = True):

    if prec != None:
        CC,ldet = prec[1]
    else:
        CC,ldet = helper_2_det(a,cov,ncov)

    if len(x.shape) == 1:
        IPROD = IP(x,CC,x)
    else:
        IPROD = IP(x,CC,x,keepdims = False)
        
#    dist = np.exp(- (1./2)*((a**2) + IPROD))*a/np.sqrt(det)#this is (unnormalized) P[a]*P[I|a]
    ldist = - (1./2)*( IPROD) - (1./2)*ldet - len(cov)*np.log(2*np.pi)/2#this is (unnormalized) P[a]*P[I|a]

    if ret_exp:
        return np.exp(ldist)
    else:
        return ldist

def PGIA(n,x,a,cov,ncov,prec = None):

    if prec != None:
        xc,xci = prec[0]
    else:
        xc,xci = helper_1_cov(a,cov,ncov)

    if len(x.shape) == 1:
        coef = np.dot(xci[n],x)/a
    else:
        coef = np.tensordot(x,xci[n],axes = [1,0])/a
        
    out = coef
    
    return out
        
def norm(n,x,a,cov,ncov,prec = None ,ret_exp = True):
    pia = PIA(n,x,a,cov,ncov,prec,ret_exp)

    if ret_exp:
        return Pa(a) * pia
    else:
        return np.log(Pa(a)) + pia
    
def integrand(n,x,a,cov,ncov,prec = None):

    pia = PIA(n,x,a,cov,ncov,prec)
    
    pgia = PGIA(n,x,a,cov,ncov,prec)
    
    out = pgia*pia*Pa(a)
    
    return out


def att_egia(n,I,a,cov,ncov,qcov,F,getall = False,getP = False):
    #we're basically going to perform 2 steps of kalman update equations starting with the prior
    #I can be a list of images, and I will write it recursively to handle any number

    out = []
    pout = []
    g = np.zeros(len(cov))
    p = cov

    for i in I:        
        g,p = att_egia_update(a,i,g,p,ncov,qcov,F)
        out.append(g)
        pout.append(p)
        g = np.dot(F,g)
        p = np.dot(F,np.dot(p,np.transpose(F))) + qcov
        
    if getP:
        return np.array(out),np.array(pout)
    
    if getall:
        return np.array(out)
    
    return out[-1][n]

def att_egia_update(a,I,g,p,ncov,q,F):
    K = a*np.dot(p,np.linalg.inv(a*a*p + ncov))

    go = g + np.dot(K,I - a*g)
    P = np.dot(np.identity(len(I)) - a*K,p)

    return go,P

def att_integrand(n,I,a,cov,ncov,qcov,F):

    pia = att_PIA_iter(I,a,cov,ncov,qcov,F)
    gia = att_egia(n,I,a,cov,ncov,qcov,F)
    
    '''
    xc,xci = helper_1_cov(a,cov,ncov)
    CC,ldet = helper_2_det(a,cov,ncov)

    if len(x.shape) == 1:
        coef = np.dot(xci[n],x)/a
        IPROD = IP(x,CC,x)
        
    else:
        coef = np.tensordot(x,xci[n],axes = [1,0])/a
        IPROD = IP(x,CC,x,keepdims = False)
        
    Ldist = - (1./2)*((a**2) + IPROD) + np.log(a) - (1./2)*ldet#this is (unnormalized) P[a]*P[I|a]

    out = coef*np.exp(ldist)
    '''
    
    return pia*gia*Pa(a)

def att_P_integrand(n,I,a,cov,ncov,qcov,F):

    pia = att_PIA_iter(I,a,cov,ncov,qcov,F)
    gia,Pia = att_egia(n,I,a,cov,ncov,qcov,F,getP = True)
    
    '''
    xc,xci = helper_1_cov(a,cov,ncov)
    CC,ldet = helper_2_det(a,cov,ncov)

    if len(x.shape) == 1:
        coef = np.dot(xci[n],x)/a
        IPROD = IP(x,CC,x)
        
    else:
        coef = np.tensordot(x,xci[n],axes = [1,0])/a
        IPROD = IP(x,CC,x,keepdims = False)
        
    Ldist = - (1./2)*((a**2) + IPROD) + np.log(a) - (1./2)*ldet#this is (unnormalized) P[a]*P[I|a]

    out = coef*np.exp(ldist)
    '''
    
    return pia*Pia[-1,n,n]*Pa(a)

def Pa(a,log = False):
    if log:
        return np.log(a) - a*a/2
    else:
        return a * np.exp(-a*a/2)

def att_norm(I,a,cov,ncov,qcov,F):

    pia = att_PIA_iter(I,a,cov,ncov,qcov,F)
    
    '''
    xc,xci = helper_1_cov(a,cov,ncov)
    CC,ldet = helper_2_det(a,cov,ncov)

    if len(x.shape) == 1:
        coef = np.dot(xci[n],x)/a
        IPROD = IP(x,CC,x)
        
    else:
        coef = np.tensordot(x,xci[n],axes = [1,0])/a
        IPROD = IP(x,CC,x,keepdims = False)
        
    ldist = - (1./2)*((a**2) + IPROD) + np.log(a) - (1./2)*ldet#this is (unnormalized) P[a]*P[I|a]
v
    out = coef*np.exp(ldist)
    '''

    
    return pia*Pa(a)

def general_att_egia(I,H,cov,ncov,qcov,F,getall = False):
    #we're basically going to perform 2 steps of kalman update equations starting with the prior
    #I can be a list of images, and I will write it recursively to handle any number

    out = []
    pout = []
    g = np.zeros(len(cov))
    p = cov
    
    
    for i in I:
        g,p = general_att_egia_update(H,i,g,p,ncov,qcov,F)

        out.append(g)
        pout.append(p)
    
    if getall:
        return np.array(out)
    
    return out[-1]

def general_att_egia_update(H,I,g,p,ncov,q,F):
    K = np.dot(p,np.dot(H.transpose(),np.linalg.inv(np.dot(H,np.dot(p,H.transpose())) + ncov)))

    go = np.dot(F,g) + np.dot(K,I - np.dot(H,g))
    P = np.dot(np.identity(len(I)) - np.dot(K,H),p)
    P = np.dot(F,np.dot(P,np.transpose(F))) + q

    return go,P

    
def helper_1_cov(a,cov,ncov):
    xc = np.identity(len(cov)) + (a**(-2))*np.dot(ncov,np.linalg.inv(cov))

    xci = np.linalg.inv(xc)

    return xc,xci

def helper_2_det(a,cov,ncov):
    CC = ncov + (a**2)*cov
    s,logdet = np.linalg.slogdet(CC)
    
    return np.linalg.inv(CC),logdet

def gexp(n,x,cov,ncov,precom = False,alow = ALOW,ahigh = AHIGH,npnt = NPT):
    
    if precom == False:
        TOP = [INT(lambda a: integrand(n,k,a,cov,ncov),0,np.inf)[0] for k in x]

        NORM = [INT(lambda a: norm(n,k,a,cov,ncov),0,np.inf)[0] for k in x]
    
        OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else 0 for k in range(len(TOP))]
    
        return np.array(OUT)
    else:
        da = (ahigh - alow)/(npnt - 1)
        pnts = np.linspace(alow,ahigh,npnt)

        prec = [[helper_1_cov(a,cov,ncov),helper_2_det(a,cov,ncov)] for a in pnts]

        TOP = np.array([integrand(n,x,pnts[i],cov,ncov,prec[i]) * da for i in range(npnt)]) #[npt,nx]
        NORM = np.array([norm(n,x,pnts[i],cov,ncov,prec[i]) * da for i in range(npnt)])
        

        TOP = TOP.sum(axis = 0)
        NORM = NORM.sum(axis = 0)

        OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else 0 for k in range(len(TOP))]

        return np.array(OUT)

def g(x,C,NC,precom = False,alow = ALOW,ahigh = AHIGH,npnt = NPT):
    return np.concatenate([np.reshape(gexp(i,x,C,NC,precom = precom,alow = alow,ahigh = ahigh,npnt = npnt),[-1,1]) for i in range(x.shape[1])],axis = 1)

def gnn(x,cov):
    '''
    Description: computes the expectation of g in the noiseless GSM

    inputs:
     x   : filter valies
     cov : covariance

    '''

    ll = LAM(x,np.linalg.inv(cov))

    n = float(cov.shape[0])
    
    return (x / np.sqrt(ll + eps))*np.float32(mpbesselk((1. - n)/2.,ll + EPS)/mpbesselk((2. - n)/2.,ll + EPS))

def gnn_MAP(x,cov):
   
    n = np.prod(x.shape[1:])
 
    ll = LAM(x,np.linalg.inv(cov))**2

    aa = np.sqrt((1. + np.sqrt(4.*ll + (n - 1.)**2) - n)/2)

    aa = np.reshape(aa,[x.shape[0],1])

    return x/aa

def MGSM_g(F,C,NC,P):
    return CC_MGSM_g(F,C[0],C[1],C[2],NC[0],NC[1],NC[2],P)


def MGSM_att_g(F1,F2,C,NC,Q,F,P):
    return CC_MGSM_att_g(F1,F2,C[0],C[1],C[2],NC[0],NC[1],NC[2],Q[0],Q[1],Q[2],F[0],F[1],F[2],P)

def MGSM_gnn(F,C,P):
    return CC_MGSM_gnn(F,C[0],C[1],C[2],P)

def CC_MGSM_gnn(F,CC,CCS,CS,P):
    '''
    Description: computes the expected g of the center filter set in the no-noise MGSM

    '''
    probs = np.transpose(get_CC_seg_weight(F,CC,CCS,CS,P))

    return (np.reshape(probs,(len(P),-1,1))*np.array([gnn(get_CC_seg_x(F,i)[0],CC if i == 0 else CCS[i-1])[:,:8] for i in range(len(P))])).sum(axis = 0)

def reverse_index(ind):
    '''
    this is a utility that takes a list of indices, and defines a new list that reverses that index swap
    '''

    out = [0 for k in range(len(ind))]

    for k in range(len(ind)):
        out[ind[k]] = k

    return out

def pos(L,x):
    '''
    return the position of x in L
    '''

    for k in range(len(L)):
        if L[k] == x:
            return k

    return False

def reverse_indices(seg,ind):
    '''
    this is a utility that takes a list of indices, and defines a new list that reverses that index swap, when not all indices are present
    '''

    out = []

    for k in range(len(ind)):
        out.append(pos(seg,ind[k]))

    return out

def get_MGSM_weights(F,segs,C,P):
    gsm_resp = np.array([P[k]*np.prod([PShared(F[:,segs[k][j]],C[k][j]) for j in range(len(segs[k]))],axis = 0) for k in range(len(segs))]) # shape [nseg,ndata]
    return np.transpose(np.squeeze(gsm_resp/np.sum(gsm_resp,axis = 0,keepdims = True)))

def general_MGSM_gnn(F,segs,C,P):

    '''
    Description:performs MGSM inference on a general MGSM model.

    args:
     F - filters [ndata,nfilt]
     segs - a list of indices for each seg [[S11...S1n1],...,[Sm1...Smnm]] where each Sij is a list of indices for the filters in the ith segment of teh mth segmentation
     C - a list of covariance matrices [[C11...Cn1]...[Cm1...Cmnm]]
     P - a list of probabilities for each seg. [P1,...,Pm]
     
    '''

    seg_to_index = [reverse_indices(np.concatenate(s),np.arange(len(np.concatenate(s)))) for s in segs]

    
    probs = np.transpose(get_MGSM_weights(F,segs,C,P))#[nseg,ndata]

    gsm_resp = np.array([np.concatenate([gnn(F[:,segs[k][j]],C[k][j]) for j in range(len(segs[k]))],axis = 1)[:,seg_to_index[k]] for k in range(len(segs))]) # shape [nseg,ndata,nfilt]

    result = np.sum(np.expand_dims(probs,-1) * gsm_resp,axis = 0)

    return result

def get_seg_indices(segs,ind):
    '''
    Description: takes a segmentation specification and a list of indices, and returns a speficication of which segments and indices within the segment must be computed to get the filters listed in ind

    
    '''
    out = []

    #I need ot go throuhg each segmentation. For each segment I need to construct an array of all the filters I need from it.

    #the result is [[11,12,...,1n],...,[m1,m2,...,mn]] where each nm item is a list itself of integers

    for k in segs:
        temp = []
        for s in range(len(k)):
            temp.append([])
            for i in range(len(k[s])):
                if k[s][i] in ind:
                    temp[-1].append(i)
        out.append(temp)
    return out                    
    
def CC_MGSM_g(F,CC,CCS,CS,NCC,NCCS,NCS,P,prec = False,npnt = NPT,alow = ALOW,ahigh = AHIGH):
    
    '''
    Description: computes the expected g of the center filter set in the noisy MGSM
    '''

    #first we need to get the assignment probabilities:
    #[[],..,[]]

    probs = np.transpose(get_noisy_CC_seg_weight(F,CC,CCS,CS,NCC,NCCS,NCS,P))
    
    gv = np.array([[gexp(n,get_CC_seg_x(F,i)[0],CC if i == 0 else CCS[i-1],NCC if i == 0 else NCCS[i-1],precom = prec,npnt = npnt,alow = alow,ahigh = ahigh) for n in range(8)] for i in range(len(P))])

    return np.transpose((np.reshape(probs,(len(P),1,-1))*gv).sum(axis = 0),(1,0))

def PShared(X,cov,mean = False,log = False):

    '''
    Description: given filters and covariances computes P[x|cov,shared]
    inputs: 
    - x - filter values - [n_data,n_site,n_ori,n_phase]
    - c - covariance - [nsite*n_ori*n_phase,nsite*n_ori*n_phase]
    '''    

    shapeX = X.shape
    shapeC = cov.shape

    if np.prod(shapeX[1:]) != shapeC[0] or np.prod(shapeX[1:]) != shapeC[1] or shapeC[0] != shapeC[1]:

        print(cov.shape)
        print(X.shape)
        raise ValueError('Covariance and data shape mismatch.')


    #flatten along the filter-location axes if it exists
    x = np.reshape(X,(shapeX[0],np.prod(shapeX[1:])))
    shape = x.shape

    lam = np.sqrt(IP(x,np.linalg.inv(cov),x))

    s,logdet = np.linalg.slogdet(cov)
    
    log_dcoef = - (shape[1]*np.log(2*math.pi) + logdet)/2.

    log_norm = (1. - float(shape[1])/2.)*np.log(lam)

    out =  log_dcoef + np.float32(mplog(mpbesselk(1. - (float(shape[1])/2.),lam +EPS))) + log_norm

    if log == False:
        out =  np.exp(out)

    if mean:
        return out.mean()

    else:
        return out

    
def LPShared(X,cov,mean = False):

    '''
    Description: given filters and covariances computes P[x|cov,shared]
    inputs: 
    - x - filter values - [n_data,n_site,n_ori,n_phase]
    - c - covariance - [nsite*n_ori*n_phase,nsite*n_ori*n_phase]
    '''    

    shapeX = X.shape
    shapeC = cov.shape

    if np.prod(shapeX[1:]) != shapeC[0] or np.prod(shapeX[1:]) != shapeC[1] or shapeC[0] != shapeC[1]:

        print(cov.shape)
        print(X.shape)
        raise ValueError('Covariance and data shape mismatch.')


    #flatten along the filter-location axes if it exists
    x = np.reshape(X,(shapeX[0],np.prod(shapeX[1:])))
    shape = x.shape

    lam = np.sqrt(IP(x,np.linalg.inv(cov),x))

    s,logdet = np.linalg.slogdet(cov)
    
    log_dcoef = - (shape[1]*np.log(2*math.pi) + logdet)/2

    log_norm = (1. - float(shape[1])/2)*np.log(lam)

    out =  np.exp(log_dcoef + np.float32(mplog(mpbesselk(1 - (float(shape[1])/2),lam+EPS))) + log_norm)

    if mean:
        return out.mean()

    else:
        return out

t_C = T.matrix("cov","float32")
DDS_var = T.nlinalg.det(t_C)*T.nlinalg.MatrixInverse()(t_C)
DDS = theano.function([t_C],DDS_var,allow_input_downcast = True)

def np_DDS(C):
    return np.linalg.det(C)*np.linalg.inv(C)

t_x = T.matrix("vec_in","float32")
XDX_var = (t_x * T.tensordot(t_x,T.nlinalg.MatrixInverse()(t_C),axes = [1,0])).sum(axis = 1)
XDX = theano.function([t_C,t_x],XDX_var,allow_input_downcast = True)

def np_XDX(C,x):
    return (x*np.tensordot(x,np.linalg.inv(C),axes = [1,0])).sum(axis = 1)

def att_LAM(C,Q,F,x):
    first_term = (x[:,0]*np.tensordot(x[:,0],np.linalg.inv(C),axes = [1,0])).sum(axis = 1)

    Tn = x[:,1:]
    Tnp1 = np.tensordot(x[:,:-1,:],F,axes = [2,1])

    dif = Tn - Tnp1
    
    other_terms = dif*np.tensordot(dif,np.linalg.inv(Q),axes = [2,0])
    other_terms = (other_terms).sum(axis = (1,2))

    out = first_term + other_terms

    if np.any(out < 0):
        print(Q)
        print(F)
        print(np.min(first_term))
        print(np.max(first_term))
        print(np.min(other_terms))
        print(np.max(other_terms))
        print("lam")
        exit()
        
    return rectify(first_term + other_terms),dif

OP_var = t_x.dimshuffle([0,'x',1])*t_x.dimshuffle([0,1,'x'])
TI_var = T.transpose(T.nlinalg.MatrixInverse()(t_C))

DXDX_var = - T.tensordot(T.tensordot(OP_var,TI_var,axes = [1,1]),TI_var,axes = [1,0])
DXDX = theano.function([t_C,t_x],DXDX_var,allow_input_downcast = True)

def np_DXDX(C,x,prec = []):
    TI = np.transpose(np.linalg.inv(C))
    if len(prec) == 0:
        sh = x.shape
        
        A = np.reshape(x,[sh[0],sh[1],1])
        B = np.reshape(x,[sh[0],1,sh[1]])
        
        OP = A*B
    else:
        OP = prec
    
    return -np.tensordot(np.tensordot(OP,TI,axes = [1,1]),TI,axes = [1,0])

def np_DXDX_att(C,x):
    TI = np.transpose(np.linalg.inv(C))
    sh = x.shape
    A = np.reshape(x,[sh[0],-1,sh[2],1])
    B = np.reshape(x,[sh[0],-1,1,sh[2]])
    
    OP = (A*B).sum(axis = 1)
    
    return -np.tensordot(np.tensordot(OP,TI,axes = [1,1]),TI,axes = [1,0])

def n(C):
    return float(len(C))

def D(C):
    return np.linalg.det(C)

###
t_A = T.tensor3("A","float32")
t_B = T.matrix("B","float32")

t_Bs = t_B.shape
t_As = t_A.shape

t_aa = t_A.dimshuffle([0,1,2,'x'])
t_bb = t_B.dimshuffle(['x','x',0,1])

t_chsum = (t_bb * (t_aa.dimshuffle([0,1,3,2]) + t_aa.dimshuffle([0,2,3,1]))).sum(axis = 3)

chsum = theano.function([t_A,t_B],t_chsum,allow_input_downcast = True)

def np_chsum(A,B):

    Bs = B.shape
    As = A.shape

    bb = np.reshape(B,[Bs[0],Bs[1],1,1]).transpose([2,3,0,1])
    aa = np.reshape(A,[As[0],As[1],As[2],1])

    o1 = (bb * aa.transpose([0,1,3,2])).sum(axis = 3)
    o2 = (bb * aa.transpose([0,2,3,1])).sum(axis = 3)

    out1 = o1 + o2

    return out1

def CHtov(ch):
    CH = np.array(ch)
    return np.reshape(np.concatenate([CH[i,:i+1] for i in range(len(CH))]),[-1])

def vtoCH(v):
    L = len(v)

    vt = np.array(v)

    Z = np.zeros(L)

    return np.array([np.concatenate([vt[(i)*(i+1)/2:(i+1)*(i+2)/2],Z[:int((np.sqrt(1 + 8*L)/2) - 1 - (i))]]) for i in range(int((np.sqrt(1 + 8*L) - 1)/2))])

def IDchol(n,array = False):
    if array:
        return np.reshape(np.concatenate([[1. if k == i else 0. for k in range(i+1)] for i in range(n)]),[-1])
    else:
        return np.reshape(np.concatenate([[1. if k == i else 0. for k in range(i+1)] for i in range(n)]),[-1]).tolist()

def CtoS(M):
   return np.dot(np.transpose(M),M)

def vtoS(v):
    return CtoS(vtoCH(v))

def LP_Q_grad():
    '''
    The only change between this and LPgrad is that lambda = I1 CC I1 + (I2 - F.I1).Q.(I2-F.I1) + ... + (In - F.In-1).Q.(In - F.In-1)
    So, the Q gradient will be the same except with a sum of terms for each term in lambda, and the F gradient will similarly be the same up to the last layer in the chain rule, where it will have a sum of terms (and the terms will be of a different strucutre)

    F grad ~ In-1 Q T F In-1 - 2 In Q T In-1 (T denotes outer product)
    Q grad ~ (In - F.In-1)T(In - F. In-1) (T here denotes outer product)

    '''
    
def LPgrad(X,v,prec = [],split = 5000):
    LL = len(X)

    if LL >= split:
        ns = int(LL/split)
        LL = split*ns
        
        X2 = np.split(X[:LL],ns)
        X1 = X[LL:]
        
        if len(prec) >0 :
            P2 = np.split(prec[:LL],len(X)/split)
            P1 = prec[LL:]
        else:
            P2 = [[] for x in X2]
            
        out = []

        for x in range(len(X2)):
            out += [f_LPgrad(X2[x],v,P2[x])]

        if len(X1) > 0:
            out += [f_LPgrad(X1,v,P1)]

            
        return np.concatenate(out)

    else:
        if len(prec) >0 :
            P1 = prec
        else:
            P1 = [[] for x in X]
        return f_LPgrad(X,v,P1)
        
def f_LPgrad(X,v,prec = []):

    '''
    This computes the gradient of PShared w.r.t. cov
    '''
    tt = time.time()
    
    x = np.reshape(X,[X.shape[0],-1])

    cho = vtoCH(v)
    cov = vtoS(v)

    N = int(n(cov))

    DT = DS(cov)

    TI = np.linalg.inv(cov)
    TIT = np.transpose(TI)

    DdetT = np.reshape(np_DDS(cov),[1,N,N])

    if len(prec) > 0:
        DxTx = np_DXDX(cov,x,prec)
    else:
        DxTx = np_DXDX(cov,x)

    xTx = np_XDX(cov,x)

    denom = 2*DT*np.float32(mpbesselk(1 - (N/2),np.sqrt(xTx) + EPS))

    num1 = np.reshape(DdetT,[1,N,N])*np.float32(mpbesselk((-2 + N)/2,np.reshape(np.sqrt(xTx),[-1,1,1])+EPS))

    num2 = DT*np.reshape(DxTx,[-1,N,N])*np.float32(mpbesselk(N/2,np.reshape(np.sqrt(xTx),[-1,1,1])+EPS))/np.reshape(np.sqrt(xTx),[-1,1,1])
    
    arg = (num1 + num2)/np.reshape(denom,[-1,1,1])

    t1 = time.time()
    temp = -chsum(arg,cho)
    t2 = time.time()

    fin = np.array([CHtov(ch) for ch in temp])
    t3 = time.time()

    return fin


def line_search(C,dc,P,ds = .1,eps = 10**-5):
    Ct = np.array(C)
    DC = np.array(dc)
    P1 = P(Ct)
    
    Ct += ds*DC

    P2 = P(Ct)

    dp = np.abs(P1 - P2)
    P1 = P2

    n = 0
    while dp > eps and n < 10000:

        n += 1

        Ct += ds*DC
        
        P2 = P(Ct)
        
        dp = np.abs(P1- P2)

        if P2 > P1:
            ds *= 1.01
        else:
            ds *= -1./2

        P1 = P2

    return Ct

def take_GSM_step(x,cchol,qchol,Fc,steps = 1,lr = .01,weights = [],fq_shared = False,f_ID = False):

    '''
    if fq shared then fix F and Q to be self-consistent according to S = F.S.F + Q

    
    '''

    if f_ID:
        fq_shared = True
        F = np.eye(x.shape[2])*sigmoid(Fc)
    else:
        F = Fc

    if weights is []:
        W = np.ones(x.shape[0])
    else:
        W = weights

    W = W/np.sum(W)

    CC = cchol
    QQ = qchol
    FF = Fc

    LR = lr

    L1 = np.mean(att_LogLikelihood(x,cchol,qchol,F,mean = False,fq_shared = fq_shared)*W,axis = 0)

    step = 0
    if f_ID:
        dc,dq,df = f_LP_att_grad(x,cchol,np.sqrt(1.-(Fc*Fc))*cchol,F,False)
    else:
        dc,dq,df = f_LP_att_grad(x,cchol,qchol,F,fq_shared)

    if f_ID:
        df = np.sum(np.diagonal(df,axis1=1,axis2=2),axis = 1) - 2 * Fc * np.sum(dq * np.expand_dims(vtoS(CC),0),axis = (1,2))
        dc += np.sqrt(1. - Fc*Fc)*dq
        
    elif fq_shared:
        df += - 2 * np.tensordot(dq,np.dot(vtoS(CC),FF),axes = [1,1])
        
    while step != steps and step < 10000:
        step += 1

        if f_ID:
            FF += np.mean(dsigmoid(df) * W)*LR
            QQ = Q_self_con(vtoS(CC),sigmoid(FF)*np.eye(len(F)))
            
        else:
            FF += np.mean(df * np.reshape(W,[-1,1,1]),axis = 0)*LR
        
        if fq_shared:
            QQ = Q_self_con(vtoS(CC),FF)
        else:
            QQ += np.mean(dq * np.reshape(W,[-1,1]),axis = 0) * LR
            
        CC += np.mean(dc * np.reshape(W,[-1,1]),axis = 0)*LR

        L2 = np.mean(att_LogLikelihood(x,CC,QQ,sigmoid(FF)*np.eye(len(F)),mean = False,fq_shared = fq_shared)*W,axis = 0)

        if L2 < L1:
            LR *= -1./2
        else:
            LR *= 1.1
        if np.abs((L2 - L1)/LR) < 1e-10:
            break
        L1 = L2

    return CC,QQ,FF

def take_FID_GSM_step(x,cchol,F,steps = 1,lr = .01,weights = []):

    '''
    if fq shared then fix F and Q to be self-consistent according to S = F.S.F + Q
    furthermore, require that F be proportional to teh identity
    '''


    Fmat = np.eye(x.shape[2])*sigmoid(F)
    fval = sigmoid(F)
    
    if weights is []:
        W = np.ones(x.shape[0])
    else:
        W = weights

    W = W/np.sum(W)

    qchol = np.sqrt(1. - fval*fval)*cchol

    LR = lr

    L1 = np.mean(att_LogLikelihood(x,cchol,qchol,Fmat,mean = False)*W,axis = 0)

    step = 0

    dc,df = f_LP_att_grad_FID(x,cchol,F)

    ccopy = np.array(cchol,copy = True)
    fcopy = np.float32(F,copy = True)

    while step != steps and step < 10000:
        step += 1

        fcopy += np.mean(df * W)*LR
        
        QQ = Q_self_con(vtoS(ccopy),sigmoid(fcopy)*np.eye(len(Fmat)))
        
        ccopy += np.mean(dc * np.reshape(W,[-1,1]),axis = 0)*LR
        
        L2 = np.mean(att_LogLikelihood(x,ccopy,QQ,sigmoid(fcopy)*np.eye(len(Fmat)),mean = False,fq_shared = True)*W,axis = 0)

        if L2 < L1:
            LR *= -1./2
        else:
            LR *= 1.1
        if np.abs((L2 - L1)/LR) < 1e-10:
            break
        L1 = L2
    return ccopy,QQ,fcopy

def fit_GSM_cov(x,INIT = [],maxsteps = 1,LS = True,weights = [],lr = .01):
    
    if len(weights) == 0:
        W = np.ones(x.shape[0]).astype("float32")
    else:
        W = np.array(weights).astype("float32")
        
    #normalize
    W = np.reshape(W/np.sum(W),[-1,1])

#    if LS == False:
#        maxsteps = 1000

    X = np.reshape(x,[x.shape[0],-1])

    nn = X.shape[1]

    if np.prod(X.shape) < nn*(nn + 1)/2:
        print("Underconstrained!")

    if len(INIT) == 0:
        Ct = IDchol(X.shape[1])
    else:
        Ct = INIT

    temp = W*PShared(x,vtoS(Ct),log = True)

    LL1 = np.sum(temp)

    XOP = np.array([np.outer(a,a) for a in X])

    for k in range(maxsteps):
        
        if LS == True:
#            dc = (W*LPgrad(X,Ct)).sum(axis = 0)
            dc = (W*LPgrad(X,Ct,XOP)).sum(axis = 0)
            Ct = line_search(Ct,dc,lambda c: np.sum(W*PShared(X,vtoS(c),log = True)))
            LL2 = np.sum(W*PShared(x,vtoS(Ct),log = True))
            
            if np.abs(LL1 - LL2) < 10**-2 or LL2 < LL1:
                break
            
            LL1 = LL2

        else:
            dc = (W*LPgrad(X,Ct,XOP)).sum(axis = 0)
            
            Ct += lr * dc

            LL2 = np.sum(W*PShared(x,vtoS(Ct),log = True))

        if np.mean(np.abs(dc)) < 10**-10:
            break

        
    return Ct

def get_seg_weight(X,CNS,CH1,CH2,P):

    '''
    Description: computes the posterior segmentation probabilities
    '''

    Pns = np.reshape(P[0]*PShared(X,CNS),[X.shape[0],-1])

    Pseg = np.transpose(np.reshape(np.asarray([P[i] * PShared(rot_filt(X,i-1)[:,:6],CH1) * PShared(rot_filt(X,i-1)[:,6:],CH2)  for i in range(1,X.shape[1])]),[-1,X.shape[0]]))

    PROB = np.concatenate([Pns,Pseg],1)
    NORM = np.sum(PROB,axis = 1,keepdims = True)

    return PROB/NORM

def NPShared(X,CNS,NC):

    def pfunc(x,a):

        s,logdet = np.linalg.slogdet(a*a*CNS + NC)
        
        out = np.log(a) -(a*a/2.) -(1./2)*IP(x,np.linalg.inv(a*a*CNS + NC),x) - (len(CNS)/2.)*np.log(2*np.pi) - logdet/2

        return np.exp(out)
        
    return np.array([INT(lambda x:pfunc(np.reshape(f,[-1]),x),0,np.inf)[0] for f in X])
               


def get_noisy_seg_weight(X,CNS,CH1,CH2,NC,NC1,NC2,P):

    '''
    Description: computes the posterior segmentation probabilities
    '''

    Pns = np.reshape(P[0]*NPShared(X,CNS,NC),[X.shape[0],1])
    Pseg = np.transpose(np.reshape(np.asarray([P[i] * NPShared(rot_filt(X,i-1)[:,:6],CH1,NC1) * NPShared(rot_filt(X,i-1)[:,6:],CH2,NC2)  for i in range(1,X.shape[1])]),[-1,X.shape[0]]))

    PROB = np.concatenate([Pns,Pseg],1)
    NORM = np.sum(PROB,axis = 1,keepdims = True)

    return PROB/NORM

def get_CC_seg_weight(X,CC,CCS,CS,P):

    '''
    Description: computes the posterior segmentation probabilities

    '''
    seg_x = get_CC_seg_x(X,0)

    Pns = np.reshape(P[0]*np.prod(np.concatenate([PShared(seg_x[0],CC)] + [PShared(seg_x[i+1],CS[i]) for i in range(4)],axis = 1),axis = 1),[X.shape[0],1])

    Pseg = []

    for k in range(4):
        seg_x = get_CC_seg_x(X,k+1)

        Pseg.append(np.reshape(P[k+1]*np.prod(np.concatenate([PShared(seg_x[0],CCS[k])] + [PShared(seg_x[i+1],CS[i])  for i in range(4)  if i != k],axis = 1),axis = 1),[X.shape[0],1]))


    Pseg = np.transpose(np.squeeze(np.array(Pseg)))

    PROB = np.concatenate([Pns,Pseg],1) 
    NORM = np.sum(PROB,1,keepdims = True)

    return PROB/NORM

def get_noisy_CC_seg_weight(X,CC,CCS,CS,NCC,NCCS,NCS,P):

    '''
    Description: computes the posterior segmentation probabilities

    '''
    seg_x = get_CC_seg_x(X,0)

    print(seg_x[0].shape)
    
    Pns = np.reshape(P[0]*np.prod(np.concatenate([np.reshape(NPShared(seg_x[0],CC,NCC),[-1,1])] + [np.reshape(NPShared(seg_x[i+1],CS[i],NCS[i]),[-1,1]) for i in range(4)],axis = 1),axis = 1),[X.shape[0],1])

    Pseg = []

    for k in range(4):
        seg_x = get_CC_seg_x(X,k+1)

        Pseg.append(np.reshape(P[k+1]*np.prod(np.concatenate([np.reshape(NPShared(seg_x[0],CCS[k],NCCS[k]),[-1,1])] + [np.reshape(NPShared(seg_x[i+1],CS[i],NCS[i]),[-1,1])  for i in range(4)  if i != k],axis = 1),axis = 1),[X.shape[0],1]))

    Pseg = np.transpose(np.squeeze(np.array(Pseg)))

    PROB = np.concatenate([Pns,Pseg],1) 
    NORM = np.sum(PROB,1,keepdims = True)

    return PROB/NORM


def get_log_seg_weight(X,CNS,CH1,CH2,P):

    '''
    Description: computes the posterior segmentation probabilities

    Pseg = Ps * P[x1] * P[x2]
    '''

    Pns = np.reshape(np.log(P[0]) + np.log(PShared(X,CNS)),[X.shape[0],-1])

#    Pseg = np.reshape(np.asarray([P[i] * PShared(rot_filt(X,i-1)[:,:6],CH1) * PShared(rot_filt(X,i-1)[:,6:],CH2)  for i in range(1,X.shape[1])]),[X.shape[0],8])

    Pseg = np.transpose(np.reshape(np.array([np.log(P[i]) + np.log(PShared(rot_filt(X,i-1)[:,:6],CH1)) + np.log(PShared(rot_filt(X,i-1)[:,6:],CH2))  for i in range(1,X.shape[1])]),[X.shape[1]-1,-1]))

    PROB = np.concatenate([Pns,Pseg],1)
    NORM = np.log(np.sum(np.exp(PROB),1,keepdims = True))
    
    return np.exp(PROB - NORM)

def get_CC_log_seg_weight(X,CC,CCS,CS,P,LOG = False):

    seg_x = get_CC_seg_x(X,0)
    seg_cov = [CC] + CS

    Pns = np.log(P[0]) + np.sum(np.log(np.concatenate([PShared(seg_x[i],seg_cov[i]) for i in range(len(seg_x))],axis = 1)),axis = 1)

#    Pns = np.log(P[0]) + np.sum(np.log(np.concatenate([PShared(seg_x[0],CC)] + [PShared(seg_x[i+1],CS[i]) for i in range(4)],axis = 1)),axis = 1)

    Pns = np.reshape(Pns,[X.shape[0],1])

    Pseg = []

    for k in range(4):
        seg_x = get_CC_seg_x(X,k+1)

        segments = [seg_x[i] for i in range(len(seg_x)) if i != k+1]
        covs = [CCS[k]] + [CS[i] for i in range(len(CS)) if i != k]

        Pseg.append(np.log(P[0]) + np.sum(np.log(np.concatenate([PShared(segments[i],covs[i]) for i in range(len(segments))],axis = 1)),axis = 1))

#        Pseg.append(np.reshape(np.log(P[k+1]) + np.sum(np.log(np.concatenate([PShared(seg_x[0],CCS[k])] + [PShared(seg_x[i+1],CS[i]) for i in range(4)  if i != k],axis = 1)),axis = 1),[X.shape[0],1]))

    Pseg = np.transpose(np.squeeze(np.array(Pseg)),[1,0])

    PROB = np.concatenate([Pns,Pseg],1)
    NORM = np.log(np.sum(np.exp(PROB),1,keepdims = True))

    if LOG:
        return PROB - NORM

    return np.exp(PROB - NORM)

def log_likelihood_center(x,C,nolog = False):
    return np.log(PShared(x,C)).mean()

def log_likelihood(x,CNS,C1,C2,P,nolog = False):
        
    SF = [get_seg_x(x,s) for s in range(1,len(P))]
    
    S0 = np.array([P[0]*PShared(x,CNS)])
    TT = np.array([P[s]*PShared(SF[s-1][0],C1)*PShared(SF[s-1][1],C2) for s in range(1,len(P))])
    
    SS = np.transpose(np.squeeze(np.concatenate([S0,TT],axis = 0)))
    
    if nolog:
        return np.sum(SS,axis = 1).mean()
    
    return np.log(np.sum(SS,axis = 1)).mean()
 

def CC_log_likelihood(x,CC,CCS,CS,P,nolog = False):
    
    #sum of Log[]

    '''
    Description: computes the posterior segmentation probabilities

    '''

    seg_x = get_CC_seg_x(x,0)

    Pns = np.reshape(P[0]*np.prod(np.concatenate([PShared(seg_x[0],CC)] + [PShared(seg_x[i+1],CS[i]) for i in range(4)],axis = 1),axis = 1),[x.shape[0],1])


    Pseg = []

    for k in range(4):
        seg_x = get_CC_seg_x(x,k+1)

        Pseg.append(np.reshape(P[k+1]*np.prod(np.concatenate([PShared(seg_x[0],CCS[k])] + [PShared(seg_x[i+1],CS[i])  for i in range(4)  if i != k],axis = 1),axis = 1),[x.shape[0],1]))


    Pseg = np.transpose(np.squeeze(np.array(Pseg)))

    PROB = np.concatenate([Pns,Pseg],1)

    return np.log(np.sum(PROB,axis = 1)).mean()


if __name__ == "__main__":

    N = 10

    x = np.random.normal(0,1,[10,5,N])#np.ones([100,5,N])
    
    f = 1.

    fval = sigmoid(f)
    cv = IDchol(N)
    qv = [k*np.sqrt(1. - fval*fval) for k in IDchol(N)]

    Fm = fval * np.eye(N)
    Cm = vtoS(cv)
    Qc = vtoS(qv)

    Qtemp = Q_self_con(Cm,Fm)

    print(np.sum(np.reshape(np.abs(Qtemp - Qc),[-1])))

    L1 = np.mean(att_LogLikelihood(x,cv,qv,Fm,mean = False),axis = 0)
    
    dc,df = f_LP_att_grad_FID(x,cv,f)

    dd = .0001

    cv = [cv[i] + dd*(1. if i == 0 else 0) for i in range(len(cv))]
    
#    f += dd

    fval = sigmoid(f)

    Fm = fval * np.eye(N)
    
#    Qtemp = Q_self_con(Cm,Fm)
    qv = [cv[i]*np.sqrt(1. - fval*fval) for i in range(len(cv))]

    L2 = np.mean(att_LogLikelihood(x,cv,qv,Fm,mean = False),axis = 0)

    print(L2 - L1)
    
#    print(np.mean(df)*dd)
    print(np.mean(dc[:,0])*dd)
