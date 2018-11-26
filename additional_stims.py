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
import utilities.log as log

ntrial = 1

CLEAN = True
PRECOMPUTE = False
NCOR = True

params = sys.argv[1]
MODEL = sys.argv[2]

dataset,freq,scale,tot,nfilt,nang,npha,fdist,samd = misc.get_parameters(sys.argv)
scale = freq /(2*math.pi)
tot = 3*freq

ULsplit = params.split("_")

pname = MODEL + "_" + ULsplit[0] + "_" + ULsplit[1] + "_" + str(scale) + params[len(ULsplit[0]) + len(ULsplit[1]) + len(ULsplit[2]) + 2:] + "LAP"

dirname = "./inference/visualGSM/" + sys.argv[1] + "{}_{}_{}_{}".format(CLEAN,MODEL,PRECOMPUTE,NCOR)

print(dataset)

print(pname)

if (len(glob.glob("./parameters/*"+pname+"_meansub.csv"))>0):
    #if the parameters are here, load them in
    print("Using existing parameters")
    P = np.loadtxt("./parameters/pha_seg_probs_"+pname+"_meansub.csv")
    CNS = np.loadtxt("./parameters/pha_CNS_"+pname+"_meansub.csv")

    fac = np.loadtxt("./parameters/pha_fac_"+pname+".csv")
    dif = np.loadtxt("./parameters/pha_dif_"+pname+".csv")

    if MODEL == "ours":
        C1 = np.loadtxt("./parameters/pha_CS1_"+pname+"_meansub.csv")
        C2 = np.loadtxt("./parameters/pha_CS2_"+pname+"_meansub.csv")

    if MODEL == "coen_cagli":
        C1 = np.loadtxt("./parameters/pha_CS1_"+pname+"_meansub.csv")

        C1 = np.reshape(C1,[4,24,24])
        C2 = np.reshape(np.loadtxt("./parameters/pha_CS2_"+pname+"_meansub.csv"),[4,16,16])


KK = [1.]
mgs = np.sqrt(np.mean([CNS[i,i] for i in range(len(CNS))]))#average std. of the filters

con = np.array([.005 * x for x in range(20)] + [.1 + .05*x for x in range(19)])
print("Starting at time {}".format(time.time()))

#this is for noise correlations

filter_maps = np.reshape(proc.get_filter_maps(nfilt,nang,npha,freq,scale,tot,fdist),[-1,2*fdist + tot,2*fdist + tot])
Ncor = np.array([[(f1*f2).sum()/np.sqrt((f1**2).sum()*(f2**2).sum()) for f2 in filter_maps] for f1 in filter_maps])    
Ncor = sim.get_normalized_NCOR(Ncor,CNS,MODEL)

##first lets do the fill-field grating at various contrasts to get triel-to trial variability and response

FITpars = [P,CNS,C1,C2]

filt = stim.make_OTUNE_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,fac,dif)
filt = np.reshape(filt,[-1,9,4,2])
filt = (filt - np.array([[[dif]]]))/(np.array([[[fac]]]))
filt = np.reshape(filt,[-1,9,8])

print("Filter Shape : {}".format(filt.shape))

print("Filter Max : {}".format(filt.max()))

Gout,nnGout = sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/ori_tuning",FITpars,Ncor,n_trial = ntrial)

plt.plot(np.linspace(0,2*np.pi,32),Gout[0,-1,:,0])
plt.savefig("./otuningtest.pdf")
