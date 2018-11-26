import math
import numpy as np
import scipy
from scipy.integrate import quad as INT

'''

this module contains functions needed for learning the MGSM and doing inference on the MGSM


'''

##helper functions

def weighted_f_cov(F,weights = None):
    S = F.shape

    return np.cov(np.transpose(np.reshape(F,(S[0],S[1]*S[2]))),aweights = weights)

def IP(x,C,y):
    if len(x.shape) == 2:
        return (x*np.transpose(np.tensordot(C,y,axes=[1,1]))).sum(axis = 1,keepdims = True)

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

##helper functiions specific to this MGSM implementation

def LAM(X,C):
    return np.sqrt(IP(X,C,X))

def H_V(v,cov):
    return (np.identity(cov.shape[0]) + (1./v**2)*np.linalg.inv(cov))

def AI(x1,x2,cov):
    return gnn(x1 + x2,cov)/(gnn(x1,cov) + gnn(x2,cov))
 
##helper functions that manipulate filters

def rot_filt(x,i):
    shape = x.shape
#    print(shape)
    temp = x[:,[0]+np.roll(np.arange(1,shape[1]),i).tolist()]
    return np.roll(temp,i * (2 * shape[2])/shape[1],2)

def get_seg_x(X,n):

    ##WARNING THIS FUNCTION ASSUMES THE 9*8 FILTER ARRANGEMENT
    shape = X.shape

    if n == 0:
        #n = 0 is the non-co-assigned one, and we pick out only the center (0th) filter bank
        A = X[:,0]
        return A
    else:
        #now we co-assign the 0th filter bank with the filters from each surround in the n-1st orientation. Note that all the cos. phases come first, then the sin. phases, hence the n - 1, and n - 1 + 4 
        A = X[:,0]#a is the first filter bank
        B = np.reshape(X[:,1:,[n-1,n-1+4]],[shape[0],-1])#this picks out all teh n01 nad n-1+4th filters from all the surround banks

        return np.concatenate((A,B),axis = 1)#this combines them. One sticky point is: are they really in teh right order? Should the re-arrangement avbove be in row-major or column-major order? This won't effect it too much qualitatively, so I am not going to fre tabout it for the moment.

def n_int_g(x,v,cov,n):

    '''

    Description: Computes the integrand of the numerator in the inference of g for the noisy MGSM
    inputs:
      x   : filter values
      v   : the scale parameter at which the integrand is being computed
      cov : the covariance of the g
      n   : the filter index to compute

    returns: 
      the numerical value of the integrand (for an integral over v)

    '''

    HI = np.linalg.inv(H_V(v,cov))
    
    LEN = len(x)
    I = np.identity(LEN)
    

    coef = np.dot(HI[n],x)
    dist = np.exp(- (1./2)*((v**2) + IP(x,I - HI,x)))/((v**(LEN - 1))*np.sqrt(np.linalg.det(H_V(v,cov))))

    return coef*dist/v

def norm_g(x,v,cov,n):

    '''

    Description: Computes the integrand of the denominator in the inference of g for the noisy MGSM
    inputs:
      x   : filter values
      v   : the scale parameter at which the integrand is being computed
      cov : the covariance of the g
      n   : the filter index to compute

    returns: 
      the numerical value of the integrand (for an integral over v)

    '''


    HI = np.linalg.inv(H_V(v,cov))

    LEN = len(x)
    I = np.identity(LEN)

    dist = np.exp(- (1./2)*(v**2 + IP(x,I - HI,x)))/((v**(LEN - 1))*np.sqrt(np.linalg.det(H_V(v,cov))))

    return dist
    
def gexp(x,cov,n):
    '''
    Description: Calculates the expectation of g given filter valies and a covariance (in the noisy GSM) numerically

    inputs: 
      x   : [nsite*nangles] filter values
      cov : [nsite*nangle,nsite*nangles] covariance of the filters
      n   : filter index to infer

    returns:
      expectation of g computed numerically
    '''

    return np.array([INT(lambda v: n_int_g(k,v,cov,n),0,np.inf)[0]/INT(lambda v: norm_g(k,v,cov,n),0,np.inf)[0] for k in x])

def gnn(x,cov):
    '''
    Description: computes the expectation of g in the noiseless GSM

    inputs:
     x   : filter valies
     cov : covariance

    '''

    ll = LAM(x,np.linalg.inv(cov))
    n = float(cov.shape[0])
    
    return (x / np.sqrt(ll))*(scipy.special.kv((1. - n)/2.,ll)/scipy.special.kv((2. - n)/2.,ll))

def MGSM_gnn(F,C1,C2,C3,P):
    '''
    Description: computes the expected g of the center filter set in the no-noise MGSM

    '''

    return (np.reshape(P,(len(P),1,1))*np.array([gnn(get_seg_x(F,i),C1 if i == 0 else C2[i-1])[:,:8] for i in range(len(P))])).sum(axis = 0)

def MGSM_g(F,C1,C2,C3,P):
    '''
    Description: computes the expected g of the center filter set in the noisy MGSM

    '''    

    return np.transpose((np.reshape(P,(len(P),1,1))*np.array([[gexp(get_seg_x(F,i),C1 if i == 0 else C2[i-1],n) for n in range(8)] for i in range(len(P))])).sum(axis = 0),(1,0))

def PShared(X,cov):

    '''
    Description: given filters and covariances computes P[x|cov,shared]

    '''

    shape = X.shape
    #flatten along hte filter-location axes
    x = np.reshape(X,(shape[0],shape[1]*shape[2]))
    shape = x.shape

    lam = np.sqrt(IP(x,cov,x))
    dcoef = 1./np.sqrt(np.power(2*math.pi,shape[1])*np.linalg.det(cov))
    
    norm = np.sqrt(np.power(lam,shape[1]-2))

    return dcoef * scipy.special.kn(1 - (shape[1]/2),lam) / norm

def get_seg_weight(X,CNS,CH1,CH2,P):

    '''
    Description: computes the posterior segmentation probabilities
    '''

    Pns = np.reshape(P[0]*PShared(X,CNS),[X.shape[0],1])

    Pseg = np.reshape(np.asarray([P[i] * PShared(rot_filt(X,i-1)[:,:6],CH1) * PShared(rot_filt(X,i-1)[:,6:],CH2)  for i in range(1,X.shape[1])]),[X.shape[0],8])

    PROB = np.concatenate([Pns,Pseg],1)
    NORM = np.sum(PROB,1,keepdims = True)
    
    return PROB/NORM

def get_log_seg_weight(X,CNS,CH1,CH2,P):

    '''
    Description: computes the posterior segmentation probabilities

    Pseg = Ps * P[x1] * P[x2]
    '''

    Pns = np.reshape(np.log(P[0]) + np.log(PShared(X,CNS)),[X.shape[0],1])

#    Pseg = np.reshape(np.asarray([P[i] * PShared(rot_filt(X,i-1)[:,:6],CH1) * PShared(rot_filt(X,i-1)[:,6:],CH2)  for i in range(1,X.shape[1])]),[X.shape[0],8])


    Pseg = np.reshape(np.asarray([np.log(P[i]) + np.log(PShared(rot_filt(X,i-1)[:,:6],CH1)) + np.log(PShared(rot_filt(X,i-1)[:,6:],CH2))  for i in range(1,X.shape[1])]),[X.shape[0],8])

    PROB = np.concatenate([Pns,Pseg],1)
    NORM = np.sum(PROB,1,keepdims = True)
    
    return np.exp(PROB - NORM)
