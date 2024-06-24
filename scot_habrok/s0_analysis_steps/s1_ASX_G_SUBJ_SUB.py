#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
from scot_habrok.load_saved_info import *
opj = os.path.join
prf_out = 'prf_no_hrf'
prf_dir = opj(derivatives_dir, prf_out)

nr_jobs = 64

sub_list = ['sub-01']
task_list = ['AS0']
batch_num = 16
roi_fit = 'all'
constraint = '--bgfs'
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
# for sub in sub_list:
#     this_dir = opj(prf_dir, sub, ses)
#     for task in task_list:    
#         prf_job_name = f'{sub}grid{task}'            
#         job="bash"
#         # job="sbatch"
#         script_path = opj(os.path.dirname(__file__),'s1_ASX_G_SUBMIT_SLURM')        
#         # Arguments to pass to HAB_G_fit.py
#         script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --prf_out {prf_out} --grid_only"
#         os.system(f'{job} {script_path} --job-name {prf_job_name} --output-dir {this_dir} --args "{script_args}"')
#         # sys.exit()

for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for task in task_list:
        
        for batch_id in np.arange(batch_num):
            prf_job_name = f'{sub}{batch_num}{task}'            
            job="bash"
            # job="sbatch"
            script_path = opj(os.path.dirname(__file__),'s1_ASX_G_SUBMIT_SLURM')        
            # Arguments to pass to HAB_G_fit.py
            script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --prf_out {prf_out} --batch_id {batch_id} --batch_num {batch_num}"

            os.system(f'{job} {script_path} --job-name {prf_job_name} --output-dir {this_dir} --args "{script_args}"')
            sys.exit()