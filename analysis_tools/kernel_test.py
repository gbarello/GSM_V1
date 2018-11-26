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
import image_processing.image_processing as proc

#miscelaneous stuff
import utilities.misc as misc
###Gather the parameters of the fit

print(sys.argv)
    
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
#    tot = 3*freq

n_cos_a = 4

###############################
    

#otherwise, fit the model.
imlist = glob.glob(BSDSloc + "*.jpg")

Clist = []

print("Measuring Filters")
for i in imlist[:10]:
    Clist.append(proc.get_phased_filter_samples(i,nfilt,nang,npha,freq,scale,tot,fdist,samd,MS = False))
    print(i + "\t{}".format(len(Clist[-1])))

#we want to sample from each one equally, so we find the list with the fewest entries
mlen = min([len(c) for c in Clist])

#randomise the list and cocnatenate them all into one list
Clist = np.array([c[np.random.choice(range(len(c)),mlen)] for c in Clist])
Clist = np.array([k for c in Clist for k in c])

#    dif = np.array([[np.median(Clist[:,:,k,0]),np.median(Clist[:,:,k,1])] for k in range(len(Clist[0,0]))])
#    fac = np.array([[IQR(Clist[:,:,k,0]),IQR(Clist[:,:,k,1])] for k in range(len(Clist[0,0]))])

dif = np.array([np.median(Clist[:,:,:,0]),np.median(Clist[:,:,:,1])])
fac = np.array([np.std(Clist[:,:,:,0]),np.std(Clist[:,:,:,1])])

print(dif)
print(fac)

#normalize the filter distributions to have standardized dist.
Clist = Clist - np.array([[dif]])
Clist = Clist / np.array([[fac]])
#    Clist = Clist - np.array([[[dif]]])
#    Clist = Clist / np.array([[[fac]]])
    
print("Number of samples: {}".format(Clist.shape[0]))
print("Mean : {} std : {}".format(np.median(Clist),np.std(Clist)))
print("Max : {}".format(np.max(np.reshape(Clist,[-1]))))
print("data shape : {}".format(Clist.shape))

kern = np.array([proc.LAPc(aa,freq,tot) for aa in np.linspace(0,np.pi,10)])

print(kern.shape)

for x in kern:
    print("{}\t{}".format(np.sum(x),np.sum(x)/np.sqrt(np.sum(x*x))))


np.savetxt("./kerntest_{}.csv".format(sys.argv[1]),np.reshape(kern,[10,-1]))
