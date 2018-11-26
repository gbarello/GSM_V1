import sys
import numpy as np
import MGSM_inference as MGSM
import math
import glob

def get_finite(G):
    b = np.reshape(G,[G.shape[0],-1])
    b = np.mean(b,axis = 1)

    mask = np.isfinite(b)

    print("Mask frac: {}".format(float(np.sum(mask))/mask.shape[0]))

    return G[mask],mask

def fit(data,model,EMsteps = 200,LOG = None,DEBUG = False):
    if model == "ours":
        return fit_ours(data,EMsteps,LOG,DEBUG)
    if model == "coen_cagli":
        return fit_CC(data,EMsteps,LOG,DEBUG)

def fit_ours(data,EMsteps = 200,LOG = None,DEBUG = False):

    """
    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     data - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.
     EMstep - the number of EM steps to take.
    """

    CS = data.shape
    
    #initialize
    P0 = 1./9
    P = np.array([P0] + [(1. - P0)/8 for i in range(8)])
    P = P/sum(P)

    CNS = np.identity(np.prod(CS[1:]))
    C1 = np.identity(6*np.prod(CS[2:]))
    C2 = np.identity(3*np.prod(CS[2:]))

    LL = MGSM.log_likelihood(data,CNS,C1,C2,P)
    LLt = LL

    if LOG != None:
        LOG.log(P)
        LOG.log("Log Likelihood : {}".format(LL))

    #run fit
    for epoch in range(EMsteps):
        print(epoch)
        #E step

        W = MGSM.get_seg_weight(data,CNS,C1,C2,P)

        #M step
        P2 = W.mean(axis = 0)
        P = P2/P2.sum()

        LLe = MGSM.log_likelihood(data,CNS,C1,C2,P)

        if LOG != None:
            LOG.log("Log Likelihood after E : {}".format(LLe))
        
        #First lets get the unsegmented covariance
        Gns = MGSM.gnn(np.reshape(data,(-1,np.prod(CS[1:]))),CNS)
        CNS = MGSM.weighted_cov(Gns,weights = W[:,0])
                
        #now I construct a list of rotated arrays (since we are making the cov. matrices rotatin inv.)
        rot_list = np.squeeze(np.array([MGSM.rot_filt(data[[c]],i) for c in range(len(data)) for i in range(8)]))
         
        #split them into the big group and little group
        r1 = rot_list[:,:6]
        r2 = rot_list[:,6:]
    
        #get the weights
        wlist = np.array([w[1+i] for w in W for i in range(8)])
        
        #get the covariance
    
        #now we compute the expected "g" for the segmentation and cov.
        
        G1 = MGSM.gnn(np.reshape(r1,(-1,np.prod(r1.shape[1:]))),C1)
        G2 = MGSM.gnn(np.reshape(r2,(-1,np.prod(r2.shape[1:]))),C2)
            
        C1 = MGSM.weighted_cov(G1,weights = wlist)
        C2 = MGSM.weighted_cov(G2,weights = wlist) 

        LLm = MGSM.log_likelihood(data,CNS,C1,C2,P)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LLm))
            LOG.log(P)

        if np.abs(LLm - LL) < 10**-10 and epoch > 10:
            break

        LL = LLm

    return P,CNS,C1,C2

def fit_CC(data,EMsteps = 200,LOG = None,DEBUG = False):

    """
    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     data - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.
     EMstep - the number of EM steps to take.
    """

    CS = data.shape
    
    #initialize
    P0 = 1./5
    P = np.array([P0] + [(1. - P0)/4 for i in range(4)])
    P = P/sum(P)

    CC = np.identity(np.prod(CS[2:]))
    CCS = [np.identity(np.prod(CS[2:]) + (CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]
    CSS = [np.identity((CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]

#    CC = np.cov(np.transpose(np.reshape(data[:,1],[CS[0],-1])))/2
#    CCS = [np.cov(np.transpose(np.concatenate([np.reshape(data[:,1],[CS[0],-1]),np.reshape(data[:,1:,i],[CS[0],-1])],axis = 1)))/2 for i in range(4)]
#    CSS = [np.cov(np.transpose(np.reshape(data[:,1:,k],[CS[0],-1])))/2 for k in range(4)]

    LL = MGSM.CC_log_likelihood(data,CC,CCS,CSS,P)
    LLt = LL

    if LOG != None:
        LOG.log(P)
        LOG.log("Log Likelihood : {}".format(LL))
    print("Log Likelihood : {}".format(LL))

    #run fit
    for epoch in range(EMsteps):
        print(epoch)
        #E step

        ##first we update the segmentation probs, and the center covariance.
        W = MGSM.get_CC_log_seg_weight(data,CC,CCS,CSS,P)

        print("WMIN : {}".format(np.max(W)))
        print("WMAX : {}".format(np.min(W)))

        #M step
        P2 = W.mean(axis = 0)
        P = P2/P2.sum()

        LLe = MGSM.CC_log_likelihood(data,CC,CCS,CSS,P)

        if LOG != None:
            LOG.log("Log Likelihood after E : {}".format(LLe))
        print("Log Likelihood after E : {}".format(LLe))
        
        #First lets get the unsegmented covariance
        Gns = MGSM.gnn(np.reshape(data[:,1],(CS[0],-1)),CC)

        CC = MGSM.weighted_cov(Gns,weights = W[:,0])

        ####Now we update the seg probs and the center-surround

        W = MGSM.get_CC_log_seg_weight(data,CC,CCS,CSS,P)

        print("WMIN : {}".format(np.max(W)))
        print("WMAX : {}".format(np.min(W)))

        #M step
        P2 = W.mean(axis = 0)
        P = P2/P2.sum()

                
        #now I construct the other ones

        #first the CS ones:
        CCS = [MGSM.weighted_cov(MGSM.gnn(MGSM.get_CC_seg_x(data,s+1)[0],CCS[s]),W[:,s + 1]) for s in range(4)]

        ####finally, the surround

        W = MGSM.get_CC_log_seg_weight(data,CC,CCS,CSS,P)

        print("WMIN : {}".format(np.max(W)))
        print("WMAX : {}".format(np.min(W)))

        #M step
        P2 = W.mean(axis = 0)
        P = P2/P2.sum()


        CSS_temp = []
        for s in range(4):
            #construct the inferred value of each surround for each segmentation
            filt = [MGSM.get_CC_seg_x(data,k)[s + 1] for k in range(5) if k != s+1]

            filt = np.concatenate(filt)
            GVAL = MGSM.gnn(filt,CSS[s])

            print("GMIN : {}".format(np.max(GVAL)))
            print("GMAX : {}".format(np.min(GVAL)))

            CSS_temp.append(MGSM.weighted_cov(GVAL,np.concatenate([W[:,k] for k in range(5) if k != s+1])))

        CSS = CSS_temp

        LLm = MGSM.CC_log_likelihood(data,CC,CCS,CSS,P)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LLm))
            LOG.log(P)

        print("Log Likelihood : {}".format(LLm))

        if np.abs(LLm - LL) < 10**-10 and epoch > 10:
            break

        LL = LLm

    return P,CC,CCS,CSS

def fit_center(data,EMsteps = 200,LOG = None,DEBUG = False):

    """
    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     data - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.
     EMstep - the number of EM steps to take.
    """

    CS = data.shape
    
    CNS = np.identity(np.prod(CS[1:]))

    LL = MGSM.log_likelihood_center(data,CNS)
    LLt = LL

    if LOG != None:
        LOG.log("Log Likelihood : {}".format(LL))


    #run fit
    for epoch in range(EMsteps):
        print(epoch)
        #E step
        
        Gns = MGSM.gnn(data,CNS)
        CNS = MGSM.weighted_cov(Gns)
                
        LL = MGSM.log_likelihood_center(data,CNS)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LL))

        if np.abs(LL - LLt) < 10**-10 and epoch > 10:
            break

        LLt = LL

    return CNS


if __name__ == "__main__":
    NN = 10000

    x = np.random.normal(0,.2,(NN,9,4,2))


    fit_CC(x)
