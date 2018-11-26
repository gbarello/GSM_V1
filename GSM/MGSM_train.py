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

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def fit(data,model,EMsteps = 50,LOG = None,DEBUG = False,GRAD = False):
    return fit_MGSM(data,EMsteps,LOG,DEBUG)

def ch_mat_to_vec(MAT):
    mat = MAT#np.transpose(MAT)
    return np.concatenate([mat[i,:i+1] for i in range(len(mat))])

def fit_general_MGSM(full_data,segs,EMreps = 20,batchsize = 500,lr = .01,varprint = 1,ngradstep = -1,buff = 5,fq_shared = True,f_ID = False):
    if f_ID:
        fq_shared = True
    print(fq_shared)
    P = np.ones(len(segs))/len(segs)

    dlen = len(full_data)
    vlen = np.min([dlen/10,500])
    
    vdat = full_data[:vlen]
    data = full_data[vlen:]
    dlen = len(data)
    
    #I need to make a C, NC, Q and F for all the segs

    chC = [[MGSM.IDchol(len(c),array = True) for c in s] for s in segs]
    chQ = [[MGSM.IDchol(len(c),array = True) for c in s] for s in segs]

    if f_ID:
        F = [[0. for c in s] for s in segs]
        fmat = [[np.eye(len(c))*sigmoid(0.) for c in s] for s in segs]
    else:
        F = [[np.eye(len(c))*.5 for c in s] for s in segs]

    C = [[MGSM.vtoS(c) for c in s] for s in chC]
    Q = [[MGSM.vtoS(c) for c in s] for s in chQ]

    if fq_shared:
        if f_ID:
            chQ = [[MGSM.Q_self_con(C[a][b],sigmoid(F[a][b])*np.eye(len(C[a][b]))) for b in range(len(F[a]))] for a in range(len(F))]
            Q = chQ            
        else:
            chQ = [[MGSM.Q_self_con(C[a][b],F[a][b]) for b in range(len(F[a]))] for a in range(len(F))]
            Q = chQ

    allind = np.arange(len(data))
    if f_ID:
        LL1 = MGSM.att_MGSM_loglik(vdat,segs,chC,chQ,fmat,P,fq_shared)
    else:
        LL1 = MGSM.att_MGSM_loglik(vdat,segs,chC,chQ,F,P,fq_shared)
    print("Var. LogLik: {}".format(LL1))
    bb = 0


    logout = []
    for step in range(EMreps):
        #get a batch
        
        ind = np.random.choice(allind,batchsize)
        d = data[ind]
        
        #get the seg weights
        segweights = MGSM.att_PShared_nonoise(data,segs,chC,chQ,F,P,fq_shared,f_ID)
       
        P = np.mean(segweights,axis = 0)
        #update all the covariances and such

        segweights = segweights[ind]

        for s in range(len(segs)):
            for c in range(len(segs[s])):
                if f_ID:
                    cc,qq,ff = MGSM.take_FID_GSM_step(d[:,:,segs[s][c]],chC[s][c],F[s][c],steps = ngradstep,lr = lr,weights = segweights[:,s])
                else:
                    cc,qq,ff = MGSM.take_GSM_step(d[:,:,segs[s][c]],chC[s][c],chQ[s][c],F[s][c],steps = ngradstep,lr = lr,weights = segweights[:,s],fq_shared = fq_shared,f_ID = f_ID)

                chC[s][c] = cc
                chQ[s][c] = qq
                F[s][c] = ff
                
        C = [[MGSM.vtoS(c) for c in s] for s in chC]

        if fq_shared:
            Q = chQ
        else:
            Q = [[MGSM.vtoS(c) for c in s] for s in chQ]

            
        if f_ID:
            fmat = [[sigmoid(F[k][j])*np.eye(len(Q[k][j])) for j in range(len(F[k]))] for k in range(len(F))]

            LL2 = MGSM.att_MGSM_loglik(vdat,segs,chC,chQ,fmat,P,fq_shared)
            
        else:
            LL2 = MGSM.att_MGSM_loglik(vdat,segs,chC,chQ,F,P,fq_shared)
            
        logout.append([step,LL2])
        #print the varification data LL
        if step % varprint == 0:
            llstr="{}\tVar. LogLik: {}".format(step,LL2)
            print(llstr)
            print("\t{}".format(P))

        if LL2 < LL1:
            bb += 1
        else:
            bb = 0
            LL1 = LL2

        if bb == buff:
            break


    if f_ID:
        F = [[sigmoid(F[i][j])*np.eye(len(C[i][j])) for j in range(len(F[i]))] for i in range(len(F))]
    
    return C,Q,F,P,logout
        
def fit_MGSM(dataT,EMsteps = 100,LOG = None,DEBUG = False,batchsize = 500):

    """
    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     dataT - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.
     EMstep - the number of EM steps to take.

    """

    CS = dataT.shape

    #get val data
    dlen = len(dataT)
    vdat = dataT[:dlen/10]
    dataT = dataT[dlen/10:]
    dlen = len(dataT)
    
    #initialize
    P = np.array([1.,1.,1.,1.,1.])
    P = P/sum(P)

    CC = MGSM.IDchol(np.prod(CS[2:]))
    CCS = [MGSM.IDchol(np.prod(CS[2:]) + (CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]
    CSS = [MGSM.IDchol((CS[1] - 1)*np.prod(CS[3:])) for k in range(4)]

    LL = MGSM.CC_log_likelihood(vdat,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)
    LLt = LL

    if LOG != None:
        LOG.log(P)
        LOG.log("Log Likelihood : {}".format(LL))
    else:
        print("Log Likelihood : {}".format(LL))

    #run fit
    for epoch in range(EMsteps):
        data = dataT[np.random.choice(len(dataT),batchsize)]
        
        print("Epoch: {}".format(epoch))
        #E step

        ##first we update the segmentation probs, and the center covariance.
        W = MGSM.get_CC_log_seg_weight(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)

        #M step
        P2 = W.mean(axis = 0) + Peps
        P = P2/P2.sum()

        LLe = MGSM.CC_log_likelihood(vdat,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)
        if LOG != None:
            LOG.log("Log Likelihood after E : {}".format(LLe))
        else:
            print("Log Likelihood after E : {}".format(LLe))

        #First lets get the unsegmented covariance

        CC = MGSM.fit_GSM_cov(np.reshape(data[:,0],(batchsize,-1)),INIT = CC,weights = W[:,0])
        
        ####Now we update the seg probs and the center-surround
        W = MGSM.get_CC_log_seg_weight(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)
        
        #now I construct the other ones        
        CCS = [MGSM.fit_GSM_cov(MGSM.get_CC_seg_x(data,s+1)[0],INIT = CCS[s],weights = W[:,s + 1]) for s in range(4)]
                   
        ####finally, the surround alone
        CSS_temp = []
        for s in range(4):
            W = MGSM.get_CC_log_seg_weight(data,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)
            #construct the inferred value of each surround for each segmentation
            filt = [MGSM.get_CC_seg_x(data,k)[s + 1] for k in range(5) if k != s+1]            
            filt = np.concatenate(filt,axis = 0)
            CSS_temp.append(MGSM.fit_GSM_cov(filt,INIT = CSS[s],weights = np.concatenate([W[:,k] for k in range(5) if k != s+1])))

        CSS = CSS_temp

        LLm = MGSM.CC_log_likelihood(vdat,MGSM.vtoS(CC),[MGSM.vtoS(c) for c in CCS],[MGSM.vtoS(c) for c in CSS],P)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LLm))
            LOG.log(P)
        else:
            print("Log Likelihood : {}".format(LLm))

        if np.abs(LLm - LL) < 10**-5 and epoch > 10:
            break

        LL = LLm

    return P,MGSM.vtoS(CC),np.array([MGSM.vtoS(c) for c in CCS]),np.array([MGSM.vtoS(c) for c in CSS])
    
def fit_GSM(dataT,EMsteps = 1000,LOG = None,DEBUG = False,GRAD = False,init = [],ndatsam = 1000):

    """
    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     data - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.
     EMstep - the number of EM steps to take.
    """

    CS = dataT.shape

    vCNS = MGSM.IDchol(np.prod(CS[1:]))
    CNS = MGSM.vtoS(vCNS)

    if init != []:
        vCNS = init
        CNS = MGSM.vtoS(vCNS)

    dlen = len(dataT)
    var = dataT[:dlen/10]
    dataT = dataT[dlen/10:]
    dlen = len(dataT)
        
    LL = MGSM.log_likelihood_center(var,CNS)
    LLt = LL

    if LOG != None:
        LOG.log("Log Likelihood : {}".format(LL))
    else:
        print("Log Likelihood : {}".format(LL))
        
    nb = 0
    #run fit
    lr = .01
    test = [CNS]

    for epoch in range(EMsteps):

        data = dataT[np.random.choice(dlen,ndatsam)]

        print(epoch)
        #E step

        vCNS = MGSM.fit_GSM_cov(data,INIT = vCNS,LS = False,lr = lr)
            
        CNS = MGSM.vtoS(vCNS)
                
        LL2 = MGSM.log_likelihood_center(var,CNS)
        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LL2))
        else:
            print("Log Likelihood after M : {}".format(LL2))
            
        if LL2 < LL:
            nb += 1
        else:
            LL = LL2
            nb = 0

        if (nb > 30 and epoch > 100):
            break
            
        LLt = LL

        print(LL2)
        test.append(CNS)
        
    return CNS,test


if __name__ == "__main__":

    np.random.seed(0)
    
    NN = 10000

    x = np.random.normal(0,1,[NN,1,10])

    segs = [[[0,1,2],[3,4],[5,6],[7,8,9]],[[4],[0,1,2,3],[9,8],[5,6],[7]]]

    aI = np.random.randint(0,len(segs),[NN])

    for i in range(NN):
        a = np.random.rayleigh(size = len(segs[aI[i]]))

        for k in range(len(segs[aI[i]])):
            x[i,:,segs[aI[i]][k]] *= a[k]
    
    C,Q,F,P = fit_general_MGSM(x,segs)

    print("C",C)
    print("Q",Q)
    print("F",F)
    print("P",P)
