#!/bin/bash

#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ full_COS --npnt=20 --noiseless"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ full_COS --npnt=20 --snr=.5 --TA=50"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ full_COS --npnt=20 --snr=1.0 --TA=50"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ full_COS --npnt=20 --snr=2.0 --TA=50"

#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=20 --noiseless"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=20 --snr=1.0 --TA=50"

#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --noiseless"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --snr=.5 --TA=50"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --snr=1.0 --TA=50"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --snr=1.5 --TA=50"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ stiff_carandini_rep --npnt=20 --noiseless"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ stiff_carandini_rep --npnt=20 --snr=.5 --TA=200"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ stiff_carandini_rep --npnt=20 --snr=.75 --TA=200"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ stiff_carandini_rep --npnt=20 --snr=1.0 --TA=200"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ stiff_carandini_rep --npnt=20 --snr=1.5 --TA=200"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ stiff_carandini_rep --npnt=20 --snr=2 --TA=200"

#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --noiseless"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=3.0 --TA=20"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=2.0 --TA=20"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=1.0 --TA=20"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=.75 --TA=20"
#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=.5 --TA=20"

#sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ surround_suppression --npnt=20 --snr=.5 --TA=50"
