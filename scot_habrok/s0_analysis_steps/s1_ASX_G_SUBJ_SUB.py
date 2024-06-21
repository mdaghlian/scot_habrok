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

nr_jobs = 1

sub_list = ['sub-01']
task_list = ['AS0']

roi_fit = 'all'
constraint = '--bgfs'
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for task in task_list:
        prf_job_name = f'G-{task}-{sub}'            
        job="python"
        script_path = opj(os.path.dirname(__file__),'S1_ASX_G_SUBMIT_SLURM')        
        
        # Arguments to pass to HAB_G_fit.py
        script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --prf_out {prf_out} --demo"

        os.system(f'{job} {script_path} {script_args}')
        sys.exit()