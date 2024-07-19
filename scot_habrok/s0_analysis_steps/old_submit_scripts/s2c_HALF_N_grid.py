#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
from scot_habrok.load_saved_info import *
opj = os.path.join
prf_out = 'prf_half'
hrf_version = 'old'
prf_dir = opj(derivatives_dir, prf_out)
prf_log_dir = opj(log_dir, prf_out)

sub_list = ['sub-01','sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_list = ['AS0_run-1', 'AS0_run-2', 'AS1_run-1', 'AS1_run-2', 'AS2_run-1', 'AS2_run-2']
n_jobs = 64
batch_num = 20
roi_fit = 'all'
constraint = '--tc'
ses = 'ses-1'
model = 'norm'
ow = False
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    this_log_dir = opj(prf_log_dir, sub, ses)
    if not os.path.exists(this_log_dir):
        os.makedirs(this_log_dir)    
    for task in task_list:    
        prf_job_name = f'{sub}-norm-{task}-grid'
        done_check = dag_find_file_in_folder(
            [model, roi_fit, task, 'grid', '.pkl'],
            path=this_dir,
            return_msg=None,
        )
        if (done_check is not None) & (not ow):
            print(f'Already done {done_check}')
            continue

        output_file = os.path.abspath(opj(this_log_dir, f'{prf_job_name}_OUT.txt'))
        error_file = os.path.abspath(opj(this_log_dir, f'{prf_job_name}_ERR.txt'))
        slurm_args = f'--output {output_file} --error {error_file} --job-name {prf_job_name}'
        job="bash"
        job="sbatch"
        
        script_path = opj(os.path.dirname(__file__),'HAB_N_fit_slurm')        
        # Arguments to pass to HAB_N_fit.py
        script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --n_jobs {n_jobs} {constraint} --prf_out {prf_out} --grid_only --model {model} --hrf_version {hrf_version}"
        # os.system(f'bash {script_path} --args "{script_args}"')
        os.system(f'{job} {slurm_args} {script_path} --args "{script_args}"')
        # sys.exit()
