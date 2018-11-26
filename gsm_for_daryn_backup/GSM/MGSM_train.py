import sys
import numpy as np
import MGSM_inference as MGSM
import math
import glob

Peps = 0

def get_finite(G):
    b = np.reshape(G,[G.shape[0],-1])
    b = np.mean(b,axis = 1)

    mask = np.isfinite(b)

    print("Mask frac: {}".format(float(np.sum(mask))/mask.shape[0]))

    return G[mask],mask

def fit(data,model,EMsteps = 200,LOG = None,DEBUG = False,GRAD = False):
    if model == "ours":
        return fit_ours(data,EMsteps,LOG,DEBUG,GRAD)
    if model == "coen_cagli":
        return fit_CC(data,EMsteps,LOG,DEBUG,GRAD)

def ch_mat_to_vec(MAT):
    mat = MAT#np.transpose(MAT)
    return np.concatenate([mat[i,:i+1] for i in range(len(mat))])

def fit_ours(data,EMsteps = 200,LOG = None,DEBUG = False,GRAD = False):

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

    if GRAD:
        
        vCNS = MGSM.IDchol(np.prod(CS[1:]))
        vC1 = MGSM.IDchol(6*np.prod(CS[2:]))
        vC2 = MGSM.IDchol(3*np.prod(CS[2:]))

        CNS = MGSM.vtoS(vCNS)
        C1 = MGSM.vtoS(vC1)
        C2 = MGSM.vtoS(vC2)

    else:
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
        #E step

        W = MGSM.get_log_seg_weight(data,CNS,C1,C2,P)

        #M step
        P2 = W.mean(axis = 0)
        P = P2/P2.sum()

        LLe = MGSM.log_likelihood(data,CNS,C1,C2,P)

        if LOG != None:
            LOG.log("Log Likelihood after E : {}".format(LLe))
        
        #First lets get the unsegmented covariance
        if GRAD:
            vCNS = MGSM.fit_GSM_cov(np.reshape(data,(-1,np.prod(CS[1:]))),vCNS,weights = W[:,0])
            CNS = MGSM.vtoS(vCNS)
        else:
            Gns = MGSM.gnn(np.reshape(data,(-1,np.prod(CS[1:]))),CNS)
            CNS = MGSM.weighted_cov(Gns,weights = W[:,0])
                
        #now I construct a list of rotated arrays (since we are making the cov. matrices rotatin inv.)
        rot_list = np.squeeze(np.array([MGSM.rot_filt(data[[c]],i) for c in range(len(data)) for i in range(8)]))

        #split them into the big group and little group
        r1 = rot_list[:,:6]
        r2 = rot_list[:,6:]
        
        #get the weights
        wlist = np.array([w[1+i] for w in W for i in range(8)])

        rind = np.random.choice(np.arange(len(r1)),len(data))
        
        r1 = np.take(r1,rind,axis = 0)
        r2 = np.take(r2,rind,axis = 0)
        wlist = np.take(wlist,rind,axis = 0)
        #get the covariance
    
        #now we compute the expected "g" for the segmentation and cov.

        if GRAD:
            vC1 = MGSM.fit_GSM_cov(np.reshape(r1,(-1,np.prod(r1.shape[1:]))),vC1,weights = wlist) 
            vC2 = MGSM.fit_GSM_cov(np.reshape(r2,(-1,np.prod(r2.shape[1:]))),vC2,weights = wlist)

            C1 = MGSM.vtoS(vC1)
            C2 = MGSM.vtoS(vC2)
        else:
            G1 = MGSM.gnn(np.reshape(r1,(-1,np.prod(r1.shape[1:]))),C1)
            G2 = MGSM.gnn(np.reshape(r2,(-1,np.prod(r2.shape[1:]))),C2)
            
            C1 = MGSM.weighted_cov(G1,weights = wlist)
            C2 = MGSM.weighted_cov(G2,weights = wlist) 

        LLm = MGSM.log_likelihood(data,CNS,C1,C2,P)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LLm))
            LOG.log(P)

        if np.abs(LLm - LL) < 10**-2 and epoch > 10:
            break

        LL = LLm
        if DEBUG:
            print(LLm)
            print(P)

    return P,CNS,C1,C2

def fit_CC(data,EMsteps = 200,LOG = None,DEBUG = False,GRAD = False):

    """

    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     data - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.
     EMstep - the number of EM steps to take.

    """

    CS = data.shape
    
    #initialize
    P0 = .1
    P = np.array([P0] + [(1. - P0)/4 for i in range(4)])
    P = P/sum(P)

    if GRAD == False:
        CC = np.identity(np.prod(CS[2:]))
        CCS = [np.identity(np.prod(CS[2:]) + (CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]
        CSS = [np.identity((CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]
    else:
        CC = MGSM.IDchol(np.prod(CS[2:]))
        CCS = [MGSM.IDchol(np.prod(CS[2:]) + (CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]
        CSS = [MGSM.IDchol((CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]

    if GRAD == False:
        LL = MGSM.CC_log_likelihood(data,CC,CCS,CSS,P)
    else:
        LL = MGSM.CC_log_likelihood(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)

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

        if GRAD == False:
            W = MGSM.get_CC_log_seg_weight(data,CC,CCS,CSS,P)
        else:
            W = MGSM.get_CC_log_seg_weight(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)

        #M step

        P2 = W.mean(axis = 0) + Peps
        P = P2/P2.sum()

        if GRAD == False:
            LLe = MGSM.CC_log_likelihood(data,CC,CCS,CSS,P)
        else:
            LLe = MGSM.CC_log_likelihood(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)

        if LOG != None:
            LOG.log("Log Likelihood after E : {}".format(LLe))
        print("Log Likelihood after E : {}".format(LLe))


        #First lets get the unsegmented covariance

        if GRAD == False:
            Gns = MGSM.gnn_MAP(np.reshape(data[:,0],(CS[0],-1)),CC)
            CC = MGSM.weighted_cov(Gns,weights = W[:,0])
        else:
            CC = MGSM.fit_GSM_cov(np.reshape(data[:,0],(CS[0],-1)),INIT = CC,weights = W[:,0])
            
        
        ####Now we update the seg probs and the center-surround
        if GRAD == False:
            W = MGSM.get_CC_log_seg_weight(data,CC,CCS,CSS,P)
        else:
            W = MGSM.get_CC_log_seg_weight(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)
        
        #now I construct the other ones        
        if GRAD == False:
            CCS = [MGSM.weighted_cov(MGSM.gnn_MAP(MGSM.get_CC_seg_x(data,s+1)[0],CCS[s]),W[:,s + 1]) for s in range(4)]
        else:
            CCS = [MGSM.fit_GSM_cov(MGSM.get_CC_seg_x(data,s+1)[0],INIT = CCS[s],weights = W[:,s + 1]) for s in range(4)]
                   
        ####finally, the surround alone
        CSS_temp = []
        for s in range(4):

            if GRAD == False:
                W = MGSM.get_CC_log_seg_weight(data,CC,CCS,CSS,P)
            else:
                W = MGSM.get_CC_log_seg_weight(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)

            #construct the inferred value of each surround for each segmentation
            filt = [MGSM.get_CC_seg_x(data,k)[s + 1] for k in range(5) if k != s+1]
            
            filt = np.concatenate(filt,axis = 0)
            if GRAD == False:
                GVAL = MGSM.gnn_MAP(filt,CSS[s])
                
                CSS_temp.append(MGSM.weighted_cov(GVAL,np.concatenate([W[:,k] for k in range(5) if k != s+1])))
            else:
                CSS_temp.append(MGSM.fit_GSM_cov(filt,INIT = CSS[s],weights = np.concatenate([W[:,k] for k in range(5) if k != s+1])))

        CSS = CSS_temp

        if GRAD == False:
            LLm = MGSM.CC_log_likelihood(data,CC,CCS,CSS,P)
        else:
            LLm = MGSM.CC_log_likelihood(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LLm))
            LOG.log(P)

        print("Log Likelihood : {}".format(LLm))

        if np.abs(LLm - LL) < 10**-2 and epoch > 10:
            break

        LL = LLm

    if GRAD == False:
        return P,CC,CCS,CSS
    else:
        return P,MGSM.vtoS(CC),np.array([MGSM.vtoS(c) for c in CCS]),np.array([MGSM.vtoS(c) for c in CSS])
    
def fit_center(data,EMsteps = 200,LOG = None,DEBUG = False,GRAD = False,init = []):

    """
    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     data - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.
     EMstep - the number of EM steps to take.
    """

    CS = data.shape

    if GRAD == False:
        CNS = np.identity(np.prod(CS[1:]))
    else:
        vCNS = MGSM.IDchol(np.prod(CS[1:]))
        CNS = MGSM.vtoS(vCNS)
        
#    print(CNS)
        

    dlen = len(data)

    var = data[:dlen/10]
    data = data[dlen/10:]
        
    LL = MGSM.log_likelihood_center(var,CNS)
    LLt = LL

    
    if LOG != None:
        LOG.log("Log Likelihood : {}".format(LL))
    nb = 0
    #run fit
    for epoch in range(EMsteps):
        print(epoch)
        #E step

        if GRAD == False:
            Gns = MGSM.gnn_MAP(data,CNS)
            CNS = MGSM.weighted_cov(Gns)            
        else:
            vCNS = MGSM.fit_GSM_cov(data,INIT = vCNS)
            CNS = MGSM.vtoS(vCNS)
                
        LL2 = MGSM.log_likelihood_center(var,CNS)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LL2))

        if LL2 < LL:
            nb += 1
        else:
            LL = LL2
            nb = 0

        if (np.abs(LL - LLt) < 10**-5 and epoch > 100) or (nb > 10):
            break

        LLt = LL

    return CNS


if __name__ == "__main__":
    NN = 10000

    x = np.random.normal(0,.2,(NN,9,4,2))

    fit_ours(x,DEBUG = True,GRAD = True)
