import argparse

parser = argparse.ArgumentParser()

#filter parameters
parser.add_argument("dir",type = str,help = "Directory of the model to run.")
args = vars(parser.parse_args())

import utilities as utils
import model_tools

direc = args["dir"]

data = model_tools.get_model_data(direc)

for p in data["params"].keys():
    print("{}\t{}".format(p,data["params"][p]))
