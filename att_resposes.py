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
import utilities.misc as misc
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import utilities.log as log
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

#make a directory that encodes the setup in the name
dirname = "./inference/visualGSM/" + sys.argv[1] + "{}_{}_{}_{}".format(CLEAN,MODEL,PRECOMPUTE,NCOR)

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
        print("parameters not found")
        exit()
    
#now we need to compute the conversion from X to SNR
#what I want to compute is sigma (the std. of noise), cuz then I can just divide everything by it.
#we have that SNR = r/sig, If we take c0 to be the contrast at which SNR = 1 then
#sig = r[c0]
#take SNR = 1 to be 2% contrast

mgs = np.sqrt(np.mean([CNS[i,i] for i in range(len(CNS))]))#average std. of the filters
KK = np.array([.1,.5,1.,2.,10.])#SNR values
KK = np.array([2.])#SNR values

con = np.array([.005 * x for x in range(20)] + [.1 + .05*x for x in range(19)])
LOG.log("Starting at time {}".format(time.time() - T0))

#this is for noise correlations

filter_maps = np.reshape(proc.get_filter_maps(nfilt,nang,npha,freq,scale,tot,fdist),[-1,2*fdist + tot,2*fdist + tot])

if NCOR == True:
    Ncor = np.array([[(f1*f2).sum()/np.sqrt((f1**2).sum()*(f2**2).sum()) for f2 in filter_maps] for f1 in filter_maps])
else:
    Ncor = np.identity(len(filter_maps))
    
Ncor = sim.get_normalized_NCOR(Ncor,CNS,MODEL)

Qfac = float(sys.argv[5])

FITpars = [P,CNS,C1,C2]

Qcor = [Qfac*CNS,Qfac*C1,Qfac*C2,Qfac*CNS]

def self_con(Q):
        return [MGSM.F_self_con(CNS,Q[0]),[MGSM.F_self_con(C1[k],Q[1][k]) for k in range(4)],[MGSM.F_self_con(C2[k],Q[2][k]) for k in range(4)]]

def identity_like(x):
        return 1.*np.identity(len(x))

##first lets do the fill-field grating at various contrasts to get triel-to trial variability and response

#sim.make_att_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/att_size_tuning",FITpars,Ncor,Qcor,n_trial = ntrial)

#########################################

LOG.log("Surround Suppression")

filt = stim.make_SS_filters([1.],nfilt,nang,npha,freq,scale,tot,fdist)

temp = filt.shape

filt = np.reshape(filt,[-1,9,4,2])
print(filt[:,0,0].max())
filt = (filt)/(np.array([[fac]]))
filt = np.reshape(filt,[-1,9,8])

print("Filter Shape : {}".format(filt.shape))

t = .5
QQ = [np.sqrt(t) * q for q in Qcor]
sim.make_att_resp(filt,filt,MODEL,CLEAN,KK,mgs,dirname + "/no_att_size_tuning_{}".format(Qfac),FITpars,Ncor,QQ,self_con(QQ),n_trial = ntrial)

t = .2
QQ = [np.sqrt(t) * q for q in Qcor]
sim.make_att_resp(filt,filt,MODEL,CLEAN,KK,mgs,dirname + "/att_size_tuning_{}".format(Qfac),FITpars,Ncor,QQ,self_con(QQ),n_trial = ntrial)

sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/size_tuning",FITpars,Ncor,n_trial = ntrial)

exit()
LOG.log("COS")

cc = np.linspace(0,.5,20)

filt1 = stim.make_att_COS_filters(cc,0,nfilt,nang,npha,freq,scale,tot,fdist)
filt2 = stim.make_att_COS_filters(cc,1,nfilt,nang,npha,freq,scale,tot,fdist)
filt3 = stim.make_att_COS_filters(cc,2,nfilt,nang,npha,freq,scale,tot,fdist)

temp1 = filt1.shape

filt1 = np.reshape(filt1,[-1,9,4,2])
filt1 = (filt1)/(np.array([[fac]]))
filt1 = np.reshape(filt1,[-1,9,8])

temp2 = filt2.shape

filt2 = np.reshape(filt2,[-1,9,4,2])
filt2 = (filt2)/(np.array([[fac]]))
filt2 = np.reshape(filt2,[-1,9,8])

temp3 = filt3.shape

filt3 = np.reshape(filt3,[-1,9,4,2])
filt3 = (filt3)/(np.array([[fac]]))
filt3 = np.reshape(filt3,[-1,9,8])

sim.make_att_resp(filt3,filt2,MODEL,CLEAN,KK,mgs,dirname + "/s_att_COS_32".format(k),FITpars,Ncor,Qcor,Fcor,n_trial = ntrial)
sim.make_att_resp(filt3,filt3,MODEL,CLEAN,KK,mgs,dirname + "/s_att_COS_33".format(k),FITpars,Ncor,Qcor,Fcor,n_trial = ntrial)
sim.make_att_resp(filt3,filt1,MODEL,CLEAN,KK,mgs,dirname + "/s_att_COS_31".format(k),FITpars,Ncor,Qcor,Fcor,n_trial = ntrial)

sim.make_att_resp(filt2,filt2,MODEL,CLEAN,KK,mgs,dirname + "/s_att_COS_22".format(k),FITpars,Ncor,Qcor,Fcor,n_trial = ntrial)
sim.make_att_resp(filt2,filt3,MODEL,CLEAN,KK,mgs,dirname + "/s_att_COS_23".format(k),FITpars,Ncor,Qcor,Fcor,n_trial = ntrial)
sim.make_att_resp(filt1,filt1,MODEL,CLEAN,KK,mgs,dirname + "/s_att_COS_21".format(k),FITpars,Ncor,Qcor,Fcor,n_trial = ntrial)

sim.make_att_resp(filt1,filt1,MODEL,CLEAN,KK,mgs,dirname + "/s_att_COS_11".format(k),FITpars,Ncor,Qcor,Fcor,n_trial = ntrial)
      
        
sim.make_resp(filt2,MODEL,CLEAN,KK,mgs,dirname + "/2_COS",FITpars,Ncor,n_trial = ntrial)
sim.make_resp(filt1,MODEL,CLEAN,KK,mgs,dirname + "/1_COS",FITpars,Ncor,n_trial = ntrial)
sim.make_resp(filt3,MODEL,CLEAN,KK,mgs,dirname + "/3_COS",FITpars,Ncor,n_trial = ntrial)

LOG.log("done")

exit()

