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
import simtools as sim

#miscelaneous stuff
import utilities.misc as misc
import utilities.log as log

def get_filter_output(func,con,nfilt,nang,npha,freq,scale,tot,fdist,fac):
    filt = func(con,nfilt,nang,npha,freq,scale,tot,fdist)

    filt = np.reshape(filt,[-1,9,4,2])
    filt = filt/(np.array([[[fac]]]))
    filt = np.reshape(filt,[-1,9,8])

    return filt

########################################################################################
########################################################################################
#############################   DONT TOUCH THIS STUFF   ################################
########################################################################################
########################################################################################

ntrial = 1

CLEAN = True
PRECOMPUTE = False
NCOR = True

params = sys.argv[1]
MODEL = sys.argv[2]

dataset,freq,scale,tot,nfilt,nang,npha,fdist,samd = misc.get_parameters(sys.argv)
scale = freq /(2*math.pi)
tot = 5*freq

ULsplit = params.split("_")

pname = MODEL + "_" + ULsplit[0] + "_" + ULsplit[1] + "_" + str(scale) + "_" + str(tot) + params[len(ULsplit[0]) + len(ULsplit[1]) + len(ULsplit[2]) + len(ULsplit[3]) + 3:] + "LAP"

dirname = "./inference/"#visualGSM/" + sys.argv[1] + "{}_{}_{}_{}".format(CLEAN,MODEL,PRECOMPUTE,NCOR)

print(dataset)

print(pname)

if (len(glob.glob("./parameters/*"+pname+"_meansub.csv"))>0):
    #if the parameters are here, load them in
    print("Using existing parameters")
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
    print("params not found")
    exit()
    
con = np.array([.005 * x for x in range(20)] + [.1 + .05*x for x in range(19)])
print("Starting at time {}".format(time.time()))

FITpars = [P,CNS,C1,C2]

KK = None
mgs = None
Ncor = None

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

#Hey Daryn! These two lines are everything you should need to adjust in this file.

'''
this is a tag for the funtion which makes your stimuli.
The ones I use are defined in "image_processing/stimuli.py".

Unfortunately the way I have it set up now you have to use identical input/output formats to what I have here
I can add more functionality (and would be happy to!) but am busy so thought I'd get you an easy, hacky, script 
that you could start playing with sooner than later.
'''
stim_function = stim.make_OTUNE_filters

'''
This is just the name of the file you want to save. It will be saved in the directory which is automatically created in the folder "/inference"
'''
save_name = "model_responses.csv"

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

filt = get_filter_output(stim_function,con,nfilt,nang,npha,freq,scale,tot,fdist,fac)

print(filt.shape)

Gout = sim.make_resp(filt,MODEL,CLEAN,KK,mgs,dirname + "/ori_tuning",FITpars,Ncor,n_trial = ntrial,NOISE=False)

np.savetxt(dirname + save_name,Gout)
