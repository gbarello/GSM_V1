#!/bin/bash

for snr in .5 .75 1. 1.5 2.
do
    #for ndt in 0.0 #.01 .02 .03 .04 .05 .06 .07 .08 .09 .15
    #do
    #python general_GSM_responses.py model_files/model_file_16/ COS_rot --n_frame=1 --TA=500 --npnt=20 --snr=$snr
	#python general_GSM_responses.py model_files/model_file_26/ size_tuning --noise_tau=$ndt --signal_tau=.1 --con=1. --dt=.1 --n_frame=20 --npnt=10 --snr=$snr
	#for st in .1 .15 .2
	#do
	#python general_GSM_responses.py model_files/model_file_16/ ori_tuning --noise_tau=$ndt --signal_tau=.1 --con=1. --dt=.01 --n_frame=100 --npnt=10 --snr=$snr
        #done
    #done
    OMP_NUM_THREADS=20 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --snr=$snr
done
