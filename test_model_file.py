import argparse

parser = argparse.ArgumentParser()

#filter parameters
parser.add_argument("dir",type = str,help = "Directory of the model to run.")
parser.add_argument("type",type = str,help = "type of stimuli to run",default = "size_tuning")
parser.add_argument("--n_frame",type = int,default = 2,help = "number of time points to use in inference.")
parser.add_argument("--snr",type = float,default = 1.,help = "SNR of the noisy model.")
parser.add_argument("--dt",type = float,default = 1.,help = "dt to run at, relative to the fit dt.")
parser.add_argument("--npnt",type = int,default = 10,help = "Number of stimuli sampling points to use.")
parser.add_argument("--fexp",action = 'store_true',default = False,help = "Use the proper rescaling of F based on matrix logs and exponents.")
args = vars(parser.parse_args())

import numpy as np
import utilities as utils
import model_tools
import image_processing.stimuli as stim
import image_processing.make_dataset as make_data
import GSM.MGSM_inference as inference

direc = args["dir"]

data = model_tools.get_model_data(direc)

for p in data["params"].keys():
    print("{}\t{}".format(p,data["params"][p]))
pars = data["params"]


f_pos = model_tools.get_f_pos(pars["filter_position"],pars["filter_distance"]*np.max(pars["wavelengths"]),pars["n_surrounds"])
print(pars.keys())
indices = np.concatenate([[[a,b,c] for a in range(pars["n_angles"]) for b in range(len(pars["wavelengths"])) for c in range(2)] for p in f_pos])
positions = np.concatenate([[p for a in range(pars["n_angles"]) for b in range(len(pars["wavelengths"])) for c in range(2)] for p in f_pos])

#now I need to make gratings, get filters, and run inference

print(np.max(f_pos))
fullsize = int(5*max(pars["wavelengths"]) + 2*np.max(f_pos))
minwav = np.min(pars["wavelengths"])

if args["type"] == "size_tuning":
    grats = [[stim.make_grating(.5,0,k,int(s),fullsize) for s in np.linspace(0,int(fullsize/2),args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "ori_tuning":
    grats = [[stim.make_grating(.5,o,k,fullsize/2,fullsize) for o in np.linspace(0,np.pi,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "CRF":
    grats = [[stim.make_grating(o,0,k,fullsize/2,fullsize) for o in np.linspace(.1,1,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "CRF_90":
    grats = [[stim.make_grating(o,np.pi/2,k,fullsize/2,fullsize) for o in np.linspace(.1,1,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "COS":
    grats = [[stim.make_grating(o,0,k,fullsize/2,fullsize) + stim.make_grating(o,np.pi/2,k,fullsize/2,fullsize) for o in np.linspace(.1,1,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "test":
    grats = [[stim.make_grating(o,0,k,fullsize/2,fullsize) for o in [.5]] for k in pars["wavelengths"][:1]]
else:
    print("Stimuli not recognized")
    exit()
    
print("Getting Coefficients")
#get filters
coeffs = np.array([[make_data.get_filter_maps(g,data["kernels"]) for g in G] for G in grats])

print(np.array(coeffs).shape)

#extract the right ones

path = [coeffs.shape[-1]/2 for k in range(args["n_frame"])]
print(path)
rundat = np.array([[make_data.sample_path(c,path,indices,positions) for c in C] for C in coeffs])/np.array([[[data["fac"]]]])

#get just the vertically oriented, cosine filters
ind = [k for k in range(len(indices)) if (indices[k][0] == 0 and positions[k][0] == 0 and positions[k][1] == 0)]
print(ind)
print("Starting Inference")
print(np.array(rundat).shape)

import scipy.linalg as linalg


if args["fexp"]:
    feps = [[linalg.logm(m)/data["params"]["walk_dt"] for m in f] for f in data["F"]]
    FF = [[np.float32(linalg.expm(args["dt"] * data["params"]["walk_dt"] * m)) for m in f] for f in feps]
else:
    FF = [[np.eye(len(m)) + args["dt"] * (m - np.eye(len(m))) for m in f] for f in data["F"]]
    
#print(FF[0][0])
QQ = [[args["dt"]*m for m in f] for f in data["Q"]]
NC = [[m/(args["snr"]**2) for m in f] for f in data["C"]]

print(rundat.shape)

test_max = [inference.find_GSM_pia_max(np.array(k),data["C"][0][0],NC[0][0],QQ[0][0],FF[0][0],0,np.inf,10.,1.) for k in rundat[0]]
test_gapp = [inference.att_egia(0,np.array(rundat[0][k]),test_max[k][0],data["C"][0][0],NC[0][0],QQ[0][0],FF[0][0]) for k in range(len(test_max))]

print(test_gapp)
print([k[0] for k in test_max])
