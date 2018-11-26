import time
import math
import numpy as np
import scipy
from scipy.integrate import quad as INT
import scipy.optimize
import scipy.linalg 
#import theano.tensor as T
#import theano

'''

this module contains functions needed for learning the MGSM and doing inference on the MGSM

'''


AHIGH = 1000
ALOW = .00001
NPT = 100000


LIMIT = 10000.

eps = .01

##helper functions

def weighted_f_cov(F,weights = None):
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

def norm(n,x,a,cov,ncov,prec = None):
    if prec != None:
        CC,ldet = prec[1]
    else:
        CC,ldet = helper_2_det(a,cov,ncov)

    if len(x.shape) == 1:
        IPROD = IP(x,CC,x)
    else:
        IPROD = IP(x,CC,x,keepdims = False)
        
#    dist = np.exp(- (1./2)*((a**2) + IPROD))*a/np.sqrt(det)#this is (unnormalized) P[a]*P[I|a]
    ldist = - (1./2)*((a**2) + IPROD) + np.log(a) - (1./2)*ldet#this is (unnormalized) P[a]*P[I|a]

    return np.exp(ldist)

def integrand(n,x,a,cov,ncov,prec = None):

    if prec != None:
        xc,xci = prec[0]
        CC,ldet = prec[1]
    else:
        xc,xci = helper_1_cov(a,cov,ncov)
        CC,ldet = helper_2_det(a,cov,ncov)

    if len(x.shape) == 1:
        coef = np.dot(xci[n],x)/a
        IPROD = IP(x,CC,x)
    else:
        coef = np.tensordot(x,xci[n],axes = [1,0])/a
        IPROD = IP(x,CC,x,keepdims = False)
    
    ldist = - (1./2)*((a**2) + IPROD) + np.log(a) - (1./2)*ldet#this is (unnormalized) P[a]*P[I|a]

    out = coef*np.exp(ldist)
    
    return out

def helper_1_cov(a,cov,ncov):
    xc = np.identity(len(cov)) + (a**(-2))*np.dot(np.linalg.inv(cov),ncov)
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
    
    return (x / np.sqrt(ll + eps))*(scipy.special.kv((1. - n)/2.,ll + eps)/scipy.special.kv((2. - n)/2.,ll + eps))

def gnn_MAP(x,cov):
   
    n = np.prod(x.shape[1:])
 
    ll = LAM(x,np.linalg.inv(cov))**2

    aa = np.sqrt((1. + np.sqrt(4.*ll + (n - 1.)**2) - n)/2)

    aa = np.reshape(aa,[x.shape[0],1])

    return x/aa

def MGSM_g(F,C1,C2,C3,NC1,NC2,NC3,P,MODEL,prec = False,npre = 5000,low = .0001,high = 600):
    if MODEL == "ours":
        return OUR_MGSM_g(F,C1,C2,C3,NC1,NC2,NC3,P,prec,npre,low,high)
    if MODEL == "coen_cagli":
        return CC_MGSM_g(F,C1,C2,C3,NC1,NC2,NC3,P,prec,npre,low,high)

def MGSM_gnn(F,C1,C2,C3,P,MODEL):
    if MODEL == "ours":
        return OUR_MGSM_gnn(F,C1,C2,C3,P)
    if MODEL == "coen_cagli":
        return CC_MGSM_gnn(F,C1,C2,C3,P)

def OUR_MGSM_gnn(F,C1,C2,C3,P):
    '''
    Description: computes the expected g of the center filter set in the no-noise MGSM

    '''
    probs = np.transpose(get_seg_weight(F,C1,C2,C3,P))

    g = np.array([gnn(get_seg_x(F,i)[0],C1 if i == 0 else C2)[:,:8] for i in range(len(P))])
    
    return (np.reshape(probs,(len(P),-1,1))*g).sum(axis = 0)

def OUR_MGSM_g(F,C1,C2,C3,NC1,NC2,NC3,P,prec = False,npnt = NPT,alow = ALOW,ahigh = AHIGH):

    '''
    Description: computes the expected g of the center filter set in the noisy MGSM

    '''    
    probs = np.transpose(get_noisy_seg_weight(F,C1,C2,C3,NC1,NC2,NC3,P))

    print(probs.shape)
    
    g = np.array([[gexp(n,get_seg_x(F,i)[0],C1 if i == 0 else C2,NC1 if i == 0 else NC2,prec,npnt,alow,ahigh) for n in range(F.shape[2])] for i in range(len(P))]) #[np,nfilt,nx]
        
    return np.transpose((np.reshape(probs,(len(P),1,-1))*g).sum(axis = 0),(1,0))

def CC_MGSM_gnn(F,CC,CCS,CS,P):
    '''
    Description: computes the expected g of the center filter set in the no-noise MGSM

    '''
    probs = np.transpose(get_CC_seg_weight(F,CC,CCS,CS,P))

    return (np.reshape(probs,(len(P),-1,1))*np.array([gnn(get_CC_seg_x(F,i)[0],CC if i == 0 else CCS[i-1])[:,:8] for i in range(len(P))])).sum(axis = 0)


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

    log_dcoef = - (shape[1]*np.log(2*math.pi) + np.trace(scipy.linalg.logm(cov)))/2

    log_norm = (1. - float(shape[1])/2)*np.log(lam)

    if log:
        out =  log_dcoef + np.log(scipy.special.kv(1 - (float(shape[1])/2),lam)) + log_norm
    else:
        out =  np.exp(log_dcoef + np.log(scipy.special.kv(1 - (float(shape[1])/2),lam)) + log_norm)

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

    log_dcoef = - (shape[1]*np.log(2*math.pi) + np.trace(scipy.linalg.logm(cov)))/2

    log_norm = (1. - float(shape[1])/2)*np.log(lam)

    out =  np.exp(log_dcoef + np.log(scipy.special.kv(1 - (float(shape[1])/2),lam)) + log_norm)

    if mean:
        return out.mean()

    else:
        return out

def np_DDS(C):
    return np.linalg.det(C)*np.linalg.inv(C)


def np_XDX(C,x):
    return (x*np.tensordot(x,np.linalg.inv(C),axes = [1,0])).sum(axis = 1)

def np_DXDX(C,x,prec = []):
    TI = np.transpose(np.linalg.inv(C))
    if len(prec) == 0:
        sh = x.shape
        
        A = np.reshape(a,[sh[0],sh[1],1])
        B = np.reshape(a,[sh[0],1,sh[1]])
        
        OP = A*B
    else:
        OP = prec
    
    return -np.tensordot(np.tensordot(OP,TI,axes = [1,1]),TI,axes = [1,0])

def n(C):
    return float(len(C))

def DS(C):
    return np.linalg.det(C)


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

def IDchol(n):
    return np.reshape(np.concatenate([[1. if k == i else 0. for k in range(i+1)] for i in range(n)]),[-1]).tolist()

def CtoS(M):
   return np.dot(np.transpose(M),M)

def vtoS(v):
    return CtoS(vtoCH(v))

def LPgrad(X,v,prec = [],split = 5000):
    LL = len(X)

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

    denom = 2*DT*scipy.special.kv(1 - (N/2),np.sqrt(xTx))

    num1 = np.reshape(DdetT,[1,N,N])*scipy.special.kv((-2 + N)/2,np.reshape(np.sqrt(xTx),[-1,1,1]))

    num2 = DT*np.reshape(DxTx,[-1,N,N])*scipy.special.kv(N/2,np.reshape(np.sqrt(xTx),[-1,1,1]))/np.reshape(np.sqrt(xTx),[-1,1,1])
    
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

def fit_GSM_cov(x,INIT = [],maxsteps = 10,LS = True,weights = []):
    
    if len(weights) == 0:
        W = np.ones(x.shape[0]).astype("float32")
    else:
        W = np.array(weights).astype("float32")
        
    #normalize
    W = np.reshape(W/np.sum(W),[-1,1])

    if LS == False:
        maxsteps = 1000

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
            
            Ct += .01 * dc

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
        return a * np.exp(-(a*a)/2) * np.exp(-(1./2)*IP(x,np.linalg.inv(a*a*CNS + NC),x))
        
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

    print(Pns.shape)
    print(Pseg.shape)

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

    x = np.random.normal(0,10,size = [25000,48])

    C = fit_GSM_cov(x,LS = True)

    exit()
    np.random.seed(1)

    NN = 10
    m = 2
    
    x = np.random.normal(0,50.,[NN,9,4,2])
    CC = np.identity(8)
    CSS = np.array([np.identity(16) for k in range(4)])
    CCS = np.array([np.identity(24) for k in range(4)])
    P = [.2,.2,.2,.2,.2]

    oCC = np.identity(72)
    oCS = np.identity(48)
    oSS = np.identity(24)

#    XX = 

#    for k in range(100):
        
        
