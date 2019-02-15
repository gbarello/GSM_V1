import utilities as utils
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

parser = argparse.ArgumentParser()

#filter parameters
parser.add_argument("--wavelengths",type = utils.csv_line(int),help = "Wavelengths in pixels of the RFs to use.",default=[16])
parser.add_argument("--n_angles",type = int,help = "number of angles between 0 and pi to use",default = 4)
parser.add_argument("--elongation",type = float,help = "Elongation of RFs.",default = 2)
parser.add_argument("--filter_position",type = str,help = "the set of position of RFs relative to center, specified by type.",default = "default")
parser.add_argument("--filter_distance",type = float,help = "the set of distance of RFs relative to center, specified by fraction of max wavelength.",default = 1.5)
parser.add_argument("--n_surrounds",type = int,help = "The number of surround filter bank locations. Organized evenly in a circle..",default = 8)
parser.add_argument("--segmentation",type = str,help = "Specify the segmentation, specified by type.",default = "default")
parser.add_argument("--total_size",type = float,help = "Total size (as mutiple of max. wavelength) to use for RF kernels.",default = 4)

#random walk params
parser.add_argument("--npaths",type = int,help = "Number of random paths to draw from each image.",default=100)
parser.add_argument("--walk_dt",type = float,help = "timescale of measurement rate. Implemented by scaling walk_std",default=1.)
parser.add_argument("--walk_std",type = float,help = "standard deviation of random walk step size in pixels",default=3.)
parser.add_argument("--walk_decay_frac",type = float,help = "Fraction of distance to decay at each timestep for random walk",default = 0.)
parser.add_argument("--walk_length",type = int,help = "Length of each random walk path.",default = 2)

#training params
parser.add_argument("--em_steps",type = int,help = "Max number of EM steps to take.",default=1000)
parser.add_argument("--fq_shared",type = int,help = "Whether to restrict F and Q to be self-consistent with the signal covariance.",default=0)
parser.add_argument("--f_ID",type = int,help = "Whether to restrict F and Q to be self-consistent with the signal covariance.",default=0)
parser.add_argument("--stochastic_buffer",type = int,help = "Number of steps with decreasing varification LL before stopping fit.",default=5)
parser.add_argument("--minibatch_size",type = int,help = "Number of data samples to use during each update.",default=1000)
parser.add_argument("--n_image",type = int,help = "Number of images to use. -1 uses all.",default=-1)
parser.add_argument("--n_grad_steps",type = int,help = "Number of adaptive-step-size gradient steps to take for each batch. Setting to -1 performs a line search for the maximum.",default=-1)
parser.add_argument("--learning_rate",type = float,help = "Initial learning rate (adjusted adaptively) for gradient ascent.",default = .01)

#misc
parser.add_argument("--seed",type = int,help = "random seed.",default=0)

args = vars(parser.parse_args())

if args["segmentation"] == "extended":
        args["filter_position"] = "extended"

if args["segmentation"] == "line":
        args["filter_position"] = "line"

args["fq_shared"] = bool(args["fq_shared"])
args["f_ID"] = bool(args["f_ID"])

import numpy as np
np.set_printoptions(precision=3)

import glob

#image processing stuff
import image_processing.make_dataset as make_data
import model_tools

#MGSM stuf
import GSM.MGSM_train as TRAIN
import simtools as sim

def IQR(dist):
        return np.percentile(dist, 75) - np.percentile(dist, 25)


#get the BSDS file location
F = open("./CONFIG","r")
for l in F:
    BSDSloc = l.split("=")[1]
    break
F.close()

img_names = glob.glob(BSDSloc + "*.jpg")[:args["n_image"]]
###########################

#set seed, extract parameters
np.random.seed(args["seed"])

nang = args["n_angles"]
wave = args["wavelengths"]

tots = args["total_size"] * np.max(wave)
elon = args["elongation"]
f_pos = model_tools.get_f_pos(args["filter_position"],args["filter_distance"]*np.max(wave),args["n_surrounds"])
npaths = args["npaths"]
dx_var = args["walk_dt"]*(args["walk_std"])**2
walk_decay_coef = np.power(args["walk_decay_frac"],args["walk_dt"])
length = args["walk_length"]
segmentation = model_tools.get_segmentation(args["segmentation"],nang,wave,f_pos)

#gather the dataset

DATA,paths,kernels = make_data.sample_images(img_names,nang, wave, tots, elon, f_pos, npaths, dx_var, walk_decay_coef, length)
DATA = np.reshape(DATA,[DATA.shape[0]*DATA.shape[1],DATA.shape[2],DATA.shape[3]])
fac = np.array([IQR(DATA[:,:,i]) for i in range(DATA.shape[-1])])

DATA = DATA / np.array([[fac]])

print("Path Std.: {}".format(np.std(paths[:,:,0]-paths[:,:,-1])))
print("Data Shape : {}".format(DATA.shape))
print("Median : {} std : {}".format(np.median(DATA),np.std(DATA)))
print("Max : {}".format(np.max(np.reshape(DATA,[-1]))))
print("Min : {}".format(np.min(np.reshape(DATA,[-1]))))
print("IQR : {}".format(IQR(np.reshape(DATA,[-1]))))

np.random.shuffle(DATA)

#run the fit 
C,Q,F,P,LOUT = TRAIN.fit_general_MGSM(DATA,segmentation,EMreps = args["em_steps"],batchsize = args["minibatch_size"],lr = args["learning_rate"],ngradstep = args["n_grad_steps"],buff = args["stochastic_buffer"],fq_shared = args["fq_shared"],f_ID = args["f_ID"])

#once it is complete, make the directory and save the data
direc = utils.get_directory(direc="./model_files/",tag = "model_file")

np.savetxt(direc + "/fac.csv",fac)
np.savetxt(direc + "/train_log.csv",LOUT)
utils.save_dict(direc + "/parameters",args)
utils.dump_file(direc + "/paths.pkl",paths)
utils.dump_file(direc + "/segs.pkl",segmentation)
utils.dump_file(direc + "/kernels.pkl",kernels)

utils.dump_file(direc + "/C.pkl",C)
utils.dump_file(direc + "/Q.pkl",Q)
utils.dump_file(direc + "/F.pkl",F)
utils.dump_file(direc + "/P.pkl",P)
