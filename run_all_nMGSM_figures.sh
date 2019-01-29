#!/bin/bash

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=50 --noiseless"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=50 --snr=.5 --TA=100"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=50 --snr=1.0 --TA=100"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=50 --snr=2.0 --TA=100"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=50 --noiseless"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=50 --snr=1.0 --TA=100"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=50 --noiseless"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=50 --snr=1.0 --TA=100"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file_3/ size_tuning --npnt=50 --noiseless"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file_3/ size_tuning --npnt=50 --snr=.75 --TA=100"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file_3/ size_tuning --npnt=50 --snr=2.0 --TA=100"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file_3/ surround_suppression --npnt=50 --noiseless"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file_3/ surround_suppression --npnt=50 --snr=.75 --TA=100"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file_3/ surround_suppression --npnt=50 --snr=1. --TA=100"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file_3/ surround_suppression --npnt=50 --snr=2.0 --TA=100"
