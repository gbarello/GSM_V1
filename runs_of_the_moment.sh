#!/bin/bash

for snr in .1 .15 2 .25 .5 
do
    for ndt in .01 .02 .03 .04 .05 .06 .07 .08 .09 .15
    do
	python general_GSM_responses.py model_files/MGSM_model_file_3/ size_tuning --noise_tau=$ndt --signal_tau=.1 --con=.5 --dt=.01 --n_frame=100 --npnt=10 --snr=$snr
	#python general_GSM_responses.py model_files/model_file_26/ size_tuning --noise_tau=$ndt --signal_tau=.1 --con=1. --dt=.1 --n_frame=20 --npnt=10 --snr=$snr
	#for st in .1 .15 .2
	#do
	#python general_GSM_responses.py model_files/model_file_16/ ori_tuning --noise_tau=$ndt --signal_tau=.1 --con=1. --dt=.01 --n_frame=100 --npnt=10 --snr=$snr
        #done
    done
done
