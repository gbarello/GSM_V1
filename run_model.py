import sys
sys.path.append('~/data')
sys.path.append('~/my_modules')
import argparse
import utilities as utils

parser = argparse.ArgumentParser()

parser.add_argument("--data",default = "BSDS")
parser.add_argument("--model",default = "GSM")
parser.add_argument("--filter_scale",default = "16")
parser.add_argument("--filter_distance",default = "16")
parser.add_argument("--n_filters",default = "4")

args = vars(parser.parse_args())

print(args)

import GSM.GSM_class as GSM

if args["model"] == "GSM":
    model = GSM.GSM()
elif args["model"] == "MGSM":
    model = GSM.MGSM()
else:
    print("Model not recognized")
    exit()


    
