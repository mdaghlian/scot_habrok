#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
from scot_habrok.load_saved_info import *
opj = os.path.join

'''
Run grid gaussian fit using default HRF [1, 4.6, 0]
'''

# SLURM ARGUMENTS (partition & profile included in the task)
sl_nodes = '1'
sl_task_per_node = '30'
sl_time = '5:00:00'

# Where is it going? What HRF version is being used
prf_out = 'prf_ascot'
n_jobs = 64
batch_num = 20
roi_fit = 'all'
constraint = '--nelder'
ses = 'ses-1'
model = 'gauss'
ow = False
ow_flag = ''

prf_dir = opj(derivatives_dir, prf_out)
prf_log_dir = opj(log_dir, prf_out)
sub_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
task_list = ['AS0', 'AS1', 'AS2'] 

# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    this_log_dir = opj(prf_log_dir, sub, ses)
    if not os.path.exists(this_log_dir):
        os.makedirs(this_log_dir)
    for task in task_list:    
        prf_job_name = f'{sub}-gauss-{task}-grid'
        done_check = dag_find_file_in_folder(
            [model, roi_fit, task, 'grid', '.pkl', f'hrf-{hrf_version}'],
            path=this_dir,
            return_msg=None,
        )
        if (done_check is not None) & (not ow):
            print(f'Already done {done_check}')
            continue
        

        output_file = os.path.abspath(opj(this_log_dir, f'{prf_job_name}_OUT.txt'))
        error_file = os.path.abspath(opj(this_log_dir, f'{prf_job_name}_ERR.txt'))
        slurm_args = f'--output {output_file} --error {error_file} --job-name {prf_job_name} ' + \
            f'--time {sl_time} --nodes {sl_nodes} --ntasks-per-node {sl_task_per_node}'
        
        job="sbatch"
        slurm_path = opj(os.path.dirname(__file__),'HAB_SLURM_GENERIC')        
        script_path = opj(os.path.dirname(__file__),'HAB_G_fit.py')        
        # Arguments to pass to HAB_G_fit.py
        script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --n_jobs {n_jobs} " + \
            f"{constraint} --prf_out {prf_out} --grid_only {ow_flag}"
        
        # os.system(f'{job} {slurm_args} {slurm_path} --script_path {script_path} --args "{script_args}"')
        os.system(f'python {script_path} {script_args}')
        sys.exit()