#!/bin/bash



for an in 0 .25 .5
do
    for as in -1 .5 1
    do 
	for snr in 1.0 1.5 2.0
	do
	    sbatch ./run_just_32.sh $snr $an $as
	done
	sbatch ./run_just_32_noiseless.sh $an $as
    done
done

exit 

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --snr=1.0 --contrast_scaling=.75 --TA=100"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --snr=1.5 --contrast_scaling=.75 --TA=100"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --snr=2 --contrast_scaling=.75 --TA=100"

exit

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --snr=1.0 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --snr=1.5 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ carandini_rep --npnt=20 --snr=2 --contrast_scaling=.75"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=20 --snr=1.0 --TA=200 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=20 --snr=1.5 --TA=200 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS --npnt=20 --snr=2.0 --TA=200 --contrast_scaling=.75"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=20 --snr=1.0 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=20 --snr=1.5 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ COS_rot --npnt=20 --snr=2.0 --contrast_scaling=.75"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --snr=1.0 --TA=50 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --snr=1.5 --TA=50 --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_18/ WTA --npnt=20 --snr=2.0 --TA=50 --contrast_scaling=.75"


sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=1.0 --contrast_scaling=.75 --TA=100 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=1.5 --contrast_scaling=.75 --TA=100 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=2.0 --contrast_scaling=.75 --TA=100 --n_frame=1"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=1.0 --contrast_scaling=.75 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=1.5 --contrast_scaling=.75 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=2.0 --contrast_scaling=.75 --n_frame=1"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ rot_surround_suppression --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ rot_surround_suppression --npnt=20 --snr=1.0 --TA=50 --contrast_scaling=.75 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ rot_surround_suppression --npnt=20 --snr=1.5 --TA=50 --contrast_scaling=.75 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ rot_surround_suppression --npnt=20 --snr=2.0 --TA=50 --contrast_scaling=.75 --n_frame=1"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=1.0 --contrast_scaling=.75 --TA=100 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=1.5 --contrast_scaling=.75 --TA=100 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/MGSM_model_file/ size_tuning --npnt=20 --snr=2.0 --contrast_scaling=.75 --TA=100 --n_frame=1"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_32/ size_tuning --npnt=20 --snr=1.0 --contrast_scaling=.75 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_32/ size_tuning --npnt=20 --snr=1.5 --contrast_scaling=.75 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_32/ size_tuning --npnt=20 --snr=2.0 --contrast_scaling=.75 --n_frame=1"

sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_32/ size_tuning --npnt=20 --noiseless --contrast_scaling=.75"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_32/ size_tuning --npnt=20 --snr=1.0 --contrast_scaling=.75 --TA=100 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_32/ size_tuning --npnt=20 --snr=1.5 --contrast_scaling=.75 --TA=100 --n_frame=1"
sbatch generic_run_script.sh "OMP_NUM_THREADS=28 python general_GSM_responses.py model_files/model_file_32/ size_tuning --npnt=20 --snr=2.0 --contrast_scaling=.75 --TA=100 --n_frame=1"

for an in 0 .25 .5
do
    for as in -1 .5 1
    do 
	for snr in 1.0 1.5 2.0
	do
	    sbatch ./run_just_32.sh $snr $an $as
	done
	sbatch ./run_just_32_noiseless.sh $an $as
    done
done
