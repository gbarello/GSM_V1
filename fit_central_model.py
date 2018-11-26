import time
import sys
import pickle
import numpy as np
import math
import glob
import os
import json
import datetime

#image processing stuff
import image_processing.test_gratings as test
import image_processing.stimuli as stim
import image_processing.image_processing as proc

#MGSM stuf
import GSM.MGSM_inference as MGSM
import GSM.MGSM_train as TRAIN
import simtools as sim

#miscelaneous stuff
import gsm_utilities.misc as misc
import gsm_utilities.log as log

import shutil

def IQR(x):
    return np.percentile(x,75) - np.percentile(x,25)

CLEAN = sys.argv[3]

if CLEAN == "True":
    CLEAN = True
else:
    CLEAN = False

if CLEAN:
    run = sys.argv[4]


NCOR = sys.argv[2]
PRECOMPUTE = True

dirname = "./inference/visualGSM/" + sys.argv[1] + "_center_" +  "{}_{}_{}".format(CLEAN,NCOR,PRECOMPUTE)

if CLEAN:
    dirname = dirname + "VAR_RUN_{}".format(run)

if os.path.exists(dirname) == False:
    os.makedirs(dirname)


DEBUG = False

T0 = time.time()

LAP = True

#get the BSDS file location
F = open("./CONFIG","r")
for l in F:
    BSDSloc = l.split("=")[1]
    break
F.close()
###########################

np.set_printoptions(precision=3)

'''

Description: this script runs the noisy and noiseless MGSM on inputs with explicit variability, in order to measure the trial-averaged response, and trial to trial variability.

'''

dataset,freq,scale,tot,nfilt,nang,npha,fdist,samd = misc.get_parameters(sys.argv)

if LAP:
    scale = freq/(2*math.pi)
    tot = 5*freq

#the number of trials to run
if CLEAN == False:
    ntrial = 1
else:
    ntrial = 1

n_cos_a = 4

params = {"data":dataset,
          "freq":freq,
          "scale":scale,
          "fsize":tot,
          "nfilt":nfilt,
          "nang":nang,
          "npha":npha,
          "fdist":fdist,
          "samd":samd,
          "clean":CLEAN,
          "precom":PRECOMPUTE,
          "lap":LAP,
          "ncor":NCOR,
          "ntrial":ntrial,
          "n_cos_a":n_cos_a}

with open(dirname + "/model_params.json", 'w') as fp:
    json.dump(params, fp)

##########################
    
pname = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
    dataset,#which dataset to use
    freq,#freq of the gabor filters (px.)
    scale,#std. of gabor gaussian (px.)
    tot,#total size off ilter kernel (px.)
    nfilt,#number of ffilters in the surround (only tested w/ 8)
    nang,#number of angles at each sampling site
    npha,#number of phases of filter 
    fdist,#distance between center and surround (px.)
    samd#distance between sampling points or filters
    )

if LAP:
    pname = pname + "LAP"

if DEBUG:
    LOG = log.log("./logs/DEBUG",name = "a debug log")
else:
    LOG = log.log("./logs/centerfit_runlog_" + pname + ".log",name = "GSM variability computation")

if DEBUG: print(pname)

if (len(glob.glob("./parameters/*"+pname+"_center.csv"))>0 and DEBUG == False):
    #if the parameters are here, load them in
    LOG.log("Using existing parameters")
    CNS = np.loadtxt("./parameters/pha_CNS_"+pname+"_center.csv")
    fac = np.loadtxt("./parameters/pha_fac_"+pname+"_center.csv")
    
else:
    #otherwise, fit the model.

    LOG.log("Parameters not found, fitting model.")

    if dataset == "BSDS":
        imlist = glob.glob(BSDSloc + "*.jpg")

    else:
        imlist = glob.glob("./images/*.jpg")
           
    Clist = []

    try:
        FF = open("./filters/test_filters_" + pname,'r')
        Clist = pickle.load(FF)
        FF.close()
        LOG.log("Using pre-saved filters")
       
    except:
        LOG.log("Measuring Filters")
        for i in imlist:
            Clist.append(proc.get_phased_filter_samples(i,nfilt,nang,npha,freq,scale,tot,fdist,samd,MS = False))
            LOG.log(i + "\t{}".format(len(Clist[-1])))

        FF = open("./filters/test_filters_" + pname,'w')
        pickle.dump(Clist,FF)
        FF.close()

    #we want to sample from each one equally, so we find the list with the fewest entries
    mlen = min([len(c) for c in Clist])

    #randomise the list and cocnatenate them all into one list
    Clist = np.array([c[np.random.choice(range(len(c)),mlen)] for c in Clist])
    Clist = np.array([k for c in Clist for k in c])

    fac = np.array([[IQR(Clist[:,:,k,0]),IQR(Clist[:,:,k,1])] for k in range(len(Clist[0,0]))])

    Clist = Clist / np.array([[fac]])

    #in this we are just going to fit a single patch, so we can flatten it 
    Clist = np.reshape(Clist,[-1,nang*npha])
    np.random.shuffle(Clist)

    A = np.cov(np.transpose(Clist))
    np.savetxt("./text_cov.csv",A)
    
    if DEBUG:
        Clist = Clist[:10000].astype('float64')
    else:
        Clist = Clist[:100000].astype('float64')

    LOG.log("Number of samples: {}".format(Clist.shape[0]))
    LOG.log("Mean : {} std : {}".format(np.median(Clist),np.std(Clist)))
    LOG.log("Max : {}".format(np.max(np.reshape(Clist,[-1]))))
    LOG.log("data shape : {}".format(Clist.shape))

    #now I need to run the EM algorithm 
    
    CNS = TRAIN.fit_center(Clist,LOG = LOG,DEBUG = DEBUG,GRAD = True)

    np.savetxt("./parameters/pha_fac_"+pname+"_center.csv",fac)
    np.savetxt("./parameters/pha_CNS_"+pname+"_center.csv",CNS)
    
#now we need to compute the conversion from X to SNR
#what I want to compute is sigma (the std. of noise), cuz then I can just divide everything by it.
#we have that SNR = r/sig, If we take c0 to be the contrast at which SNR = 1 then
#sig = r[c0]
#take SNR = 1 to be 2% contrast

print(fac)

mgs = np.sqrt(np.mean([CNS[i,i] for i in range(len(CNS))]))#average std. of the filters
KK = np.sqrt(nang/4)*mgs*np.array([.5,.75,1.,1.5,2.])#noise std. were taking it to be fixed ratios of av. filter covariance

filter_maps = np.reshape(proc.get_filter_maps(nfilt,nang,npha,freq,scale,tot,fdist,2),[9,nang*npha,2*fdist + tot,2*fdist + tot])[0]#just take the center

if NCOR:
    Ncor = np.array([[(f1*f2).sum()/np.sqrt((f1**2).sum()*(f2**2).sum()) for f2 in filter_maps] for f1 in filter_maps])
else:
    Ncor = np.identity(filter_maps.shape[0])

Ncor = CNS

Cdet = np.linalg.norm(CNS)
Ndet = np.linalg.norm(Ncor)

print(Cdet,Ndet)

Ncor = Ncor * Cdet / Ndet
KK = np.array([.5,.75,1.,1.5,2.])#noise std. were taking it to be fixed ratios of av. filter covariance

np.savetxt("./ncor"+pname+".csv",Ncor)

print("g std: {}".format(mgs))
#the number of trials to run

#con = [.01,.02,.04,.08,.16,.32,.64]
#con = [.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
con = np.array([.005 * x for x in range(20)] + [.1 + .05*x for x in range(19)])
LOG.log("Starting at time {}".format(time.time() - T0))

inittime = time.time()

print("ncor shape",Ncor.shape)


LOG.log("Cross Orientation Modulation")

#filt = stim.make_full_field_COS_filters([0] + con,nfilt,nang,npha,freq,scale,tot,fdist,4)
#filt = np.reshape(filt,[-1,9,8])

filt = stim.make_COS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,4,2)
filt = (filt)/(np.array([[fac]]))

filt = np.reshape(filt,[-1,9,npha*nang])
filt = filt[:,0,:]

print(filt.shape)
print(filt.max())

sim.make_GSM_resp(filt,CLEAN,KK,mgs,dirname + "/full_field_COS",CNS,ncor = Ncor,n_trial = ntrial)

LOG.log("Winner Takes All")

filt = stim.make_WTA_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,2)
filt = (filt)/(np.array([[fac]]))

filt = np.reshape(filt,[-1,9,npha*nang])
filt = filt[:,0,:]

print(filt.shape)
print(filt.max())

sim.make_GSM_resp(filt,CLEAN,KK,mgs,dirname + "/full_field_WTA",CNS,ncor = Ncor,n_trial = ntrial)

print(time.time() - inittime)

LOG.log("done")
