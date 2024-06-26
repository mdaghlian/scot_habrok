#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
from scot_habrok.load_saved_info import *
opj = os.path.join
prf_out = 'prf_half'
prf_dir = opj(derivatives_dir, prf_out)

n_jobs = 64

sub_list = ['sub-01','sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_list = ['AS0_run-1', 'AS0_run-2', 'AS1_run-1', 'AS1_run-2', 'AS2_run-1', 'AS2_run-2']

batch_num = 20
roi_fit = 'all'
constraint = '--tc'
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for task in task_list:    
        prf_job_name = f'{sub}Ggrid{task}'            
        # Use bash or sbatch?
        job="bash"
        job="sbatch"
        # Set variables for script
        log_dir = opj(this_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        output_file = os.path.abspath(opj(log_dir, f'OUT_{prf_job_name}.txt'))
        error_file = opj(log_dir, f'ERR_{prf_job_name}.txt')        
        slurm_args = f'--output {output_file} --error {error_file} --job-name {prf_job_name}'
        print(slurm_args)
        # sys.exit()
        script_path = opj(os.path.dirname(__file__),'HAB_G_fit_slurm_TEST')        
        # Arguments to pass to HAB_G_fit.py
        script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --n_jobs {n_jobs} {constraint} --prf_out {prf_out} --grid_only"
        os.system(f'{job} {slurm_args} {script_path} --args "{script_args}"')
        sys.exit()