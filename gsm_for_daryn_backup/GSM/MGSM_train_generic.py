import sys
import numpy as np
import MGSM_inference as MGSM
import math
import glob

#I still need to rewrite:
#
#MGSM.get_seg_probs
#MGSM.MSGM_g
#MGSM.log_likelihood
#
#
#
#
#######################

#I want to get a function that takes the data and some generic specification of the model 
#
#To do so I need: 
# - data
# - a function that generates segmentations
# - a specification of the number of segmentations
# - a specification of the number of covariance matrices
# - a specification of which segmentation uses which covariance

def fit(data,get_seg,cov_seg,EMsteps = 200,LOG = None,DEBUG = False):

    """
    Description: This takes data in and fits an MGSM model to it.

    arguments: 
     data - an array of floats w/ dim. [n_data, n_positions, ...] where ... represents any set of features that occur at each position.

     get_seg - a function which takes the data and returns all segmentatiions of the data. The returned data has shape [nseg,ndata,nsegmentations,nd_s], where nsegmentations is the number of independent mixers in a segmentation scheme, and nd_s is the number of signal variables being modulated by each mixer. For example, in the coen-cagli segmentation scheme the dimentions would be [5,ndata,x] where x = 5 for the first seg and x = 4 for the rest. The outputs should be arranged so that segmentations which share a covariance matrix are aligned.
             - for my segmentation the output should be of shape [a,:,x,y] where 
               - a:0   -> {x,y} = {{0,72}}
               - a:1-8 -> {x,y} = {{0,43},{1,24}}

     cov_seg - a set of tuples which idenftifies which covariance to use for each segmentation. Dimentions [nseg,nsegmentations].

     EMstep - the max number of EM steps to take.

     LOG - whether or not to log the output

     DEBUG - debug mode, essentially "verbose"

    """

    CS = data.shape

    #get all the segmented data
    dseg = get_seg(data)

    #this is the number of different segmentations
    nseg = len(dseg)

    #this is the labels for the cov. matrices
    cov_index = set(np.reshape(cov_seg,[-1]))

    #and this is the number of distinct cov. matrices
    ncov = len(cov_index)

    #and finally a table that tells us how large each covariance should be
    cov_size = [0 for k in range(ncov)]
    for s in range(len(cov_seg)):
        for x in range(len(cov_seg[s])):
            cov_size[cov_seg[s][x]] = len(dseg[k][0][x])

    #now I define a function that takes posterios and data and returns estimates of covariances
    def get_cov(PP,seg_data):
        '''
        Description:
         Return estimates of covariance matrices

        Arguments:
         PP - [n_seg,n_data] posterior segmentation proabilities
         seg_data - output of get_seg(data) run through an estimate for g (signal variable) [nseg,ndata,nsegmentations,nd_s]

        returns:
         COV - [n_cov] - extiamtes of the covariance matrices

        '''
        #now I need to go through each segmentation, and each data, and collect the data and the posterior
        dweights = [[] for k in range(ncov)]
        for s in range(len(seg_data)):
            for d in range(len(seg_data[s])):
                for x in range(len(seg_data[s,d])):
                    dweights[cov_seg[s,x]].append([seg_data[s,d,x],PP[s,d]])
       
        dweights = np.array(dweights)
 
        COV = [MGSM.weighted_cov(dweights[c,:,0],weights = dweights[c,:,1]) for c in range(len(dweigths))]

        return COV

    #we initialize the covariance matrices to be identity matrices
    cov = [np.identity(len(cov_size)) for k in range(len(cov_size))]

    #initialize to equal prob.
    P = np.array([1./nseg for i in range(nseg)])
    P = P/sum(P)

    LL = MGSM.log_likelihood(dseg,cov,P,cov_seg)
    LLt = LL

    if LOG != None:
        LOG.log(P)
        LOG.log("Log Likelihood : {}".format(LL))

    #run fit
    for epoch in range(EMsteps):
        print(epoch)
        #E step

        W = MGSM.get_seg_weight(dseg,cov,P,cov_seg)

        #M step
        P2 = W.mean(axis = 1)
        P = P2/P2.sum()

        LLe = MGSM.log_likelihood(dseg,cov,P,cov_seg)

        if LOG != None:
            LOG.log("Log Likelihood after E : {}".format(LLe))
        
        Gseg = MGSM.MGSM_gnn(dseg,cov,P,cov_seg)

        cov = get_cov(Gseg,W)
                
        LLm = MGSM.log_likelihood(dseg,cov,P,cov_seg)

        if LOG != None:
            LOG.log("Log Likelihood after M : {}".format(LLm))
            LOG.log(P)

        if np.abs(LLm - LL) < 10**-10 and epoch > 10:
            break

        LL = LLm

    return P,CNS,C1,C2
