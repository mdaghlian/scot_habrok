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

sub_list = ['sub-01', ] # 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_list = ['AS0',] # 'AS1', 'AS2']
batch_num = 20
roi_fit = 'all'
constraint = '--tc'
ses = 'ses-1'
model = 'norm'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for task in task_list:    
        prf_job_name = f'{sub}grid{task}'            
        job="bash"
        job="sbatch"
        script_path = opj(os.path.dirname(__file__),'HAB_N_fit_slurm')        
        # Arguments to pass to HAB_G_fit.py
        script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --prf_out {prf_out} --grid_only --model {model}"
        os.system(f'{job} {script_path} --job-name {prf_job_name} --output-dir {this_dir} --args "{script_args}"')
        sys.exit()
