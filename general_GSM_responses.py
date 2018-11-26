import time
import argparse

parser = argparse.ArgumentParser()

#filter parameters
parser.add_argument("dir",type = str,help = "Directory of the model to run.")
parser.add_argument("type",type = str,help = "type of stimuli to run",default = "size_tuning")
parser.add_argument("--n_frame",type = int,default = 2,help = "number of time points to use in inference.")
parser.add_argument("--snr",type = float,default = 1.,help = "SNR of the noisy model.")
parser.add_argument("--dt",type = float,default = 1.,help = "dt to run at, relative to the fit dt.")
parser.add_argument("--con",type = float,default = .5,help = "contrast for the stimulus.")
parser.add_argument("--npnt",type = int,default = 10,help = "Number of stimuli sampling points to use.")
parser.add_argument("--TA",type = int,default = 0,help = "If nonzero, sample with noise and this number of samples.")
parser.add_argument("--time",type = int,default = -1,help = "If nonzero, run for each number of timesteps up to {time}.")
parser.add_argument("--pad",type = int,default = 0,help = "Pad stim with {pad} zeros.")
parser.add_argument("--fexp",action = 'store_true',default = False,help = "Use the proper rescaling of F based on matrix logs and exponents.")
parser.add_argument("--variance",action = 'store_true',default = False,help = "Flag to calculate the uncertainty, P instead of the mean response.")
parser.add_argument("--tag",default = "",help = "A prefix to add to the output.")

args = vars(parser.parse_args())

import numpy as np
import utilities as utils
import model_tools
import image_processing.stimuli as stim
import image_processing.make_dataset as make_data
import GSM.MGSM_inference as inference

if args["variance"]:
    respF = inference.general_MGSM_p_att
    args["tag"] += "variance_"
else:
    respF = inference.general_MGSM_g_att

direc = args["dir"]

data = model_tools.get_model_data(direc)

for p in data["params"].keys():
    print("{}\t{}".format(p,data["params"][p]))
pars = data["params"]

def get_cross(M,ind):
    return np.array([[M[i,j] for j in ind] for i in ind])

f_pos = model_tools.get_f_pos(pars["filter_position"],pars["filter_distance"]*np.max(pars["wavelengths"]),pars["n_surrounds"])
print(pars.keys())
indices = np.concatenate([[[a,b,c] for a in range(pars["n_angles"]) for b in range(len(pars["wavelengths"])) for c in range(2)] for p in f_pos])
positions = np.concatenate([[p for a in range(pars["n_angles"]) for b in range(len(pars["wavelengths"])) for c in range(2)] for p in f_pos])

#now I need to make gratings, get filters, and run inference

print(np.max(f_pos))
fullsize = int(5*max(pars["wavelengths"]) + 2*np.max(f_pos))
minwav = np.min(pars["wavelengths"])

if args["type"] == "size_tuning":
    grats = [[stim.make_grating(args["con"],0,k,int(s),fullsize) for s in np.linspace(0,int(fullsize/2),args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "ori_tuning":
    grats = [[stim.make_grating(args["con"],o,k,fullsize/2,fullsize) for o in np.linspace(0,np.pi,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "CRF":
    grats = [[stim.make_grating(o,0,k,fullsize/2,fullsize) for o in np.linspace(.1,1,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "CRF_90":
    grats = [[stim.make_grating(o,np.pi/2,k,fullsize/2,fullsize) for o in np.linspace(.1,1,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "COS":
    grats = [[stim.make_grating(o,0,k,fullsize/2,fullsize) + stim.make_grating(o,np.pi/2,k,fullsize/2,fullsize) for o in np.linspace(.1,1,args["npnt"])] for k in pars["wavelengths"]]
elif args["type"] == "SPONT":
    grats = [[0*stim.make_grating(0,0,k,fullsize/2,fullsize)] for k in pars["wavelengths"]]
elif args["type"] == "surround_suppression":
    import image_processing.test_gratings as test

    LF = [lambda c,a,k,r,T: test.GRATC(c,a,k,r,T),
          lambda c,a,k,r,T: test.GRATC(c,a,k,r,T)
          +
          test.s_GRATC(1.,a+np.pi/2,k,r,T,surr = 0.)]

    grats = [[f(o,0,k,k*pars["filter_distance"]/2,fullsize) for o in np.logspace(-2,0,args["npnt"]) for f in LF] for k in pars["wavelengths"]]
    
elif args["type"] == "test":
    grats = [[stim.make_grating(o,0,k,fullsize/2,fullsize) for o in [.5]] for k in pars["wavelengths"][:1]]

elif args["type"] == "MI":
    import att_MGSM_responses as aresp

    RES,VRES = aresp.mutual_information_data(data,args["dt"])
    utils.dump_file(direc + "/MI_responses_{}.pkl".format(args["dt"]),RES)
    utils.dump_file(direc + "/MI_responses_variance_{}.pkl".format(args["dt"]),VRES)

    exit()
elif args["type"] == "on_off":
    import att_MGSM_responses as aresp

    dt = .5
    RES = aresp.on_off_response(data,args["snr"],dt,10)
    utils.dump_file(direc + "/on_off_responses_{}_{}.pkl".format(args["snr"],dt),RES)

    exit()
elif args["type"] == "nat_MI":
    import att_MGSM_responses as aresp

    RES = aresp.mutual_information_data(data,args["snr"],use_grat = False)

    utils.dump_file(direc + "/nat_MI_responses_{}.pkl".format(args["snr"]),RES)

    exit()
else:
    print("Stimuli not recognized")
    exit()
    
print("Getting Coefficients")
#get filters
coeffs = np.array([[make_data.get_filter_maps(g,data["kernels"]) for g in G] for G in grats])

print(np.array(coeffs).shape)

#extract the right ones

if args["time"] > -1:
    args["n_frame"] = args["time"]

path = [coeffs.shape[-1]/2 for k in range(args["n_frame"])]
print(path)
rundat = np.array([[make_data.sample_path(c,path,indices,positions) for c in C] for C in coeffs])/np.array([[[data["fac"]]]])

#get just the vertically oriented filters
ind = [k for k in range(len(indices)) if (indices[k][0] == 0 and positions[k][0] == 0 and positions[k][1] == 0)]

print(ind)
print("Starting Inference")
print(np.array(rundat).shape)

import scipy.linalg as linalg

feps = [[linalg.logm(m)/data["params"]["walk_dt"] for m in f] for f in data["F"]]
FF = [[np.float32(linalg.expm(args["dt"] * data["params"]["walk_dt"] * m)) for m in f] for f in feps]
    
#print(FF[0][0])
QQ = [[inference.Q_self_con(data["C"][k][m],FF[k][m]) for m in range(len(data["F"][k]))] for k in range(len(data["F"]))]
#QQ = [[args["dt"]*m for m in f] for f in data["Q"]]

if args["TA"]==0:
    NC = [[m/(args["snr"]**2) for m in f] for f in data["C"]]
else:
    nlist = np.zeros([len(indices),len(indices)])

    if True:
        kerprod = model_tools.get_mean_NC(data["C"],data["segs"])
        NC = [[get_cross(kerprod,b) for b in a] for a in data["segs"]]

    elif data["params"]["segmentation"] == "gsm":
        nlist = data["C"][0][0] / (args["snr"]**2)

        kerprod = nlist
        NC = [[get_cross(kerprod,b) for b in a] for a in data["segs"]]

    else:
        nlist[:len(data["segs"][0][0]),:len(data["segs"][0][0])] = data["C"][0][0]

        v1 = np.array([np.mean(np.diag(x)[::2]) for x in data["C"][0][1:]])
        v2 = np.array([np.mean(np.diag(x)[1::2]) for x in data["C"][0][1:]])
        v1 = np.diag(np.repeat(v1,[data["params"]["n_surrounds"] + 4]))/(2*args["snr"]**2)
        v2 = np.diag(np.repeat(v2,[data["params"]["n_surrounds"] + 4]))/(2*args["snr"]**2)

        nlist[len(data["segs"][0][0])::2,len(data["segs"][0][0])::2] = v1
        nlist[len(data["segs"][0][0])+1::2,len(data["segs"][0][0])+1::2] = v2


        kerprod = nlist
        NC = [[get_cross(kerprod,b) for b in a] for a in data["segs"]]

    kerprod *= 1./(args["snr"]**2)
    kerprod += np.eye(len(kerprod))*.001

t1 = time.time()

pad = np.zeros([rundat.shape[0],rundat.shape[1],args["pad"],rundat.shape[-1]])
rundat = np.concatenate([pad,rundat],axis = 2)

if args["TA"]==0:
    if args["time"]==-1:
        responses = [respF(np.array(cc),data["segs"],data["C"],NC,QQ,FF,data["P"],ind,stable = True,op=True) for cc in rundat]
        
    else:
        print("time")
        responses = []
        for t in range(1,args["time"]):
            print(t)
            responses.append([respF(np.array(cc)[:,:t],data["segs"],data["C"],NC,QQ,FF,data["P"],ind,stable = True,op=True) for cc in rundat])
else:
    
    if data["params"]["segmentation"]!= "gsm" and False:        
        print("trial averaging is not written for non GSM models")
        exit()
        
    responses = []
    for trial in range(args["TA"]):
        print(trial)

        noise = np.random.multivariate_normal(np.zeros(len(kerprod)),kerprod,np.array(rundat).shape[1:-1])

        responses.append([respF(np.array(cc) + noise,data["segs"],data["C"],NC,QQ,FF,data["P"],ind,stable = True,op=True) for cc in rundat])
    
t2 = time.time()

print("It took {} seconds.".format(t2-t1))
print("{}".format(np.sum(responses)))

if args["time"] != -1:
    args["tag"] += "t_{}".format(args["time"])
if args["TA"]==0:
    utils.dump_file(direc + "/"+args["tag"]+"responses_{}_{}_{}_{}_{}.pkl".format(args["con"],args["n_frame"],args["snr"],args["type"],args["dt"]),responses)
else:
    utils.dump_file(direc + "/"+args["tag"]+"TA_responses_{}_{}_{}_{}_{}.pkl".format(args["con"],args["n_frame"],args["snr"],args["type"],args["dt"]),responses)
utils.dump_file(direc + "/gratings.pkl",grats)
utils.dump_file(direc + "/inputs.pkl",rundat)
