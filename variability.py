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
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import gsm_utilities.log as log
import shutil
###Gather the parameters of the fit

def IQR(dist):
        return np.percentile(dist, 75) - np.percentile(dist, 25)

print(sys.argv)

if sys.argv[4] == "True":
        CLEAN = True
elif sys.argv[4] == "False":
        CLEAN = False #False means NO VARIABILITY
else:
    exit()

#MODEL = "ours"
#MODEL = "coen_cagli"
PRECOMPUTE = False
if sys.argv[3] == "True":
    NCOR = True
elif sys.argv[3] == "False":
    NCOR = False
else:
    exit()

MODEL = sys.argv[2]

R = 2

#make a directory that encodes the setup in the name
dirname = "./inference/visualGSM/" + sys.argv[1] + "{}_{}_{}_{}_{}".format(CLEAN,MODEL,PRECOMPUTE,NCOR,R)

if CLEAN == True:
    dirname += "_VAR_RUN_" + str(sys.argv[5])

if os.path.exists(dirname) == False:
    os.makedirs(dirname)
    
########################

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

Description: this script runs the noisy and noiseless MGSM on inputs with explicit variability, in order to measure the trial averaged response, and trial to trial variability.

'''

dataset,freq,scale,tot,nfilt,nang,npha,fdist,samd = misc.get_parameters(sys.argv)

if LAP:
    scale = freq /(2*math.pi)
    tot = 5*freq

#the number of trials to run
if CLEAN == False:
    ntrial = 1
else:
    ntrial = 10

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
          "model":MODEL,
          "precom":PRECOMPUTE,
          "lap":LAP,
          "ncor":NCOR,
          "ntrial":ntrial,
          "n_cos_a":n_cos_a}

with open(dirname + "/model_params.json", 'w') as fp:
    json.dump(params, fp)

###############################
    
pname = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        dataset,#which dataset to use
        freq,#freq of the gabor filters (px.)
        scale,#std. of gabor gaussian (px.)
        tot,#total size off ilter kernel (px.)
        nfilt,#number of ffilters in the surround (only tested w/ 8)
        nang,#number of angles at each sampling site
        npha,#number of phases of filter 
        fdist,#distance between center and surround (px.)
        samd,#distance between sampling points or filters
        R#ratio of x and y scales n filters
    )

pname = MODEL + "_" + pname

if LAP:
    pname = pname + "LAP"

LOG = log.log(dirname + "/variability_runlog_" + pname + ".log",name = "GSM variability computation")
print(pname)

if (len(glob.glob("./parameters/*"+pname+"_meansub.csv"))>0 and DEBUG == False):
    #if the parameters are here, load them in
    LOG.log("Using existing parameters")
    P = np.loadtxt("./parameters/pha_seg_probs_"+pname+"_meansub.csv")
    CNS = np.loadtxt("./parameters/pha_CNS_"+pname+"_meansub.csv")

    fac = np.loadtxt("./parameters/pha_fac_"+pname+".csv")

    if MODEL == "ours":
        C1 = np.loadtxt("./parameters/pha_CS1_"+pname+"_meansub.csv")
        C2 = np.loadtxt("./parameters/pha_CS2_"+pname+"_meansub.csv")

    if MODEL == "coen_cagli":
        C1 = np.loadtxt("./parameters/pha_CS1_"+pname+"_meansub.csv")

        C1 = np.reshape(C1,[4,24,24])
        C2 = np.reshape(np.loadtxt("./parameters/pha_CS2_"+pname+"_meansub.csv"),[4,16,16])

else:
    #otherwise, fit the model.

    LOG.log("Parameters not found, fitting model.")

    if dataset == "BSDS":
        imlist = glob.glob(BSDSloc + "*.jpg")

    else:
        imlist = glob.glob("./images/*.jpg")
           
    Clist = []

    try:
        FF = open("./filters/test_filters" + pname[len(MODEL):],'r')
        Clist = pickle.load(FF)
        FF.close()
        LOG.log("Using pre-saved filters")
       
    except:
        LOG.log("Measuring Filters")
        for i in imlist:
            Clist.append(proc.get_phased_filter_samples(i,nfilt,nang,npha,freq,scale,tot,fdist,samd,R,MS = False))
            LOG.log(i + "\t{}".format(len(Clist[-1])))

        FF = open("./filters/test_filters" + pname[len(MODEL):],'w')
        pickle.dump(Clist,FF)
        FF.close()

    #we want to sample from each one equally, so we find the list with the fewest entries
    mlen = min([len(c) for c in Clist])

    #randomise the list and cocnatenate them all into one list
    Clist = np.array([c[np.random.choice(range(len(c)),mlen)] for c in Clist])
    Clist = np.array([k for c in Clist for k in c])


#    fac = np.array([np.std(Clist[:,:,:,0]),np.std(Clist[:,:,:,1])])
    fac = np.array([IQR(Clist[:,:,:,0]),IQR(Clist[:,:,:,1])])

    Clist = Clist / np.array([[fac]])
    
    LOG.log("Number of samples: {}".format(Clist.shape[0]))
    LOG.log("Mean : {} std : {}".format(np.median(Clist),np.std(Clist)))
    LOG.log("Max : {}".format(np.max(np.reshape(Clist,[-1]))))
    LOG.log("IQR : {}".format(IQR(np.reshape(Clist,[-1]))))
    LOG.log("data shape : {}".format(Clist.shape))

    np.random.shuffle(Clist)
    if DEBUG:
        Clist = Clist[:100000].astype('float64')
    else:
        if fdist == 20:
            Clist = Clist[:30000].astype('float64')
        else:
            Clist = Clist[:50000].astype('float64')

    #now I need to run the EM algorithm 
   
    print("Running " + MODEL + " fit.")
 
    P,CNS,C1,C2 = TRAIN.fit(Clist,MODEL,LOG = LOG,DEBUG = DEBUG,GRAD = True)

    CNS = np.array(CNS)
    C1 = np.array(C1)
    C2 = np.array(C2)
    P = np.array(P)

    LOG.log("Probabilities:\n\t{}".format(P))

    np.savetxt("./parameters/pha_fac_"+pname+".csv",fac)

    if MODEL == "ours":
        np.savetxt("./parameters/pha_seg_probs_"+pname+"_meansub.csv",P)
        np.savetxt("./parameters/pha_CNS_"+pname+"_meansub.csv",CNS)
        np.savetxt("./parameters/pha_CS1_"+pname+"_meansub.csv",C1)
        np.savetxt("./parameters/pha_CS2_"+pname+"_meansub.csv",C2)


    if MODEL == "coen_cagli":
        np.savetxt("./parameters/pha_seg_probs_"+pname+"_meansub.csv",P)
        np.savetxt("./parameters/pha_CNS_"+pname+"_meansub.csv",CNS)
        np.savetxt("./parameters/pha_CS1_"+pname+"_meansub.csv",np.reshape(np.array(C1),[4,-1]))
        np.savetxt("./parameters/pha_CS2_"+pname+"_meansub.csv",np.reshape(np.array(C2),[4,-1]))

exit()    
#now we need to compute the conversion from X to SNR
#what I want to compute is sigma (the std. of noise), cuz then I can just divide everything by it.
#we have that SNR = r/sig, If we take c0 to be the contrast at which SNR = 1 then
#sig = r[c0]
#take SNR = 1 to be 2% contrast

mgs = np.sqrt(np.mean([CNS[i,i] for i in range(len(CNS))]))#average std. of the filters
#KK = np.array([.1,.5,1.,2.,10.])#SNR values
KK = np.array([.1,.5,1.,2.,10.])

con = np.array([.005 * x for x in range(20)] + [.1 + .05*x for x in range(19)])
#con = np.array([.9])
LOG.log("Starting at time {}".format(time.time() - T0))

#this is for noise correlations

filter_maps = np.reshape(proc.get_filter_maps(nfilt,nang,npha,freq,scale,tot,fdist,R),[-1,2*fdist + tot,2*fdist + tot])

if NCOR == True:
    Ncor = np.array([[(f1*f2).sum()/np.sqrt((f1**2).sum()*(f2**2).sum()) for f2 in filter_maps] for f1 in filter_maps])
else:
    Ncor = np.identity(len(filter_maps))
    
Ncor = sim.get_normalized_NCOR(Ncor,CNS,MODEL)

##first lets do the fill-field grating at various contrasts to get triel-to trial variability and response

FITpars = [P,CNS,C1,C2]

LOG.log("Full Field")

print(con)

filt = stim.make_full_field_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,R)
filt = np.reshape(filt,[-1,9,4,2])
filt = (filt)/(np.array([[fac]]))
filt = np.reshape(filt,[-1,9,8])

print("Filter Shape : {}".format(filt.shape))

print("Filter Max : {}".format(filt.max()))

sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/full_field",FITpars,Ncor,n_trial = ntrial)

LOG.log("Surround Suppression")

filt = stim.make_SS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,R)

temp = filt.shape

filt = np.reshape(filt,[-1,9,4,2])
print(filt[:,0,0].max())
filt = (filt)/(np.array([[fac]]))
filt = np.reshape(filt,[-1,9,8])

print("Filter Shape : {}".format(filt.shape))

sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/size_tuning",FITpars,Ncor,n_trial = ntrial)

LOG.log("More Surround Suppression")

filt = stim.make_BSS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,5,R)
filt = np.reshape(filt,[-1,9,4,2])
filt = (filt)/(np.array([[fac]]))
filt = np.reshape(filt,[-1,9,8])

sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/surr_supp",FITpars,Ncor,n_trial = ntrial)

LOG.log("Full Field COS")

filt = stim.make_full_field_COS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,4,R)
filt = np.reshape(filt,[-1,9,4,2])
filt = (filt)/(np.array([[fac]]))
filt = np.reshape(filt,[-1,9,8])

print("Filter Shape : {}".format(filt.shape))

sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/full_field_COS",FITpars,Ncor,n_trial = ntrial)

LOG.log("WTA")

filt = stim.make_WTA_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,R)
filt = np.reshape(filt,[-1,9,4,2])
filt = (filt)/(np.array([[fac]]))
filt = np.reshape(filt,[-1,9,8])

print("Filter Shape : {}".format(filt.shape))

sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/full_field_WTA",FITpars,Ncor,n_trial = ntrial)

LOG.log("done")

exit()

