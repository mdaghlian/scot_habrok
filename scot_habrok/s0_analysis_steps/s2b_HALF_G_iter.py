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
batch_num = 15
roi_fit = 'all'
constraint = '--tc'
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
i = 0
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for task in task_list:        
        for batch_id in np.arange(1, batch_num+1):
            prf_job_name = f'{sub}{batch_num}{task}'            
            i += 1
            if i>50:
                print(f'stopped at {prf_job_name}')
                sys.exit()
            # # job="bash"
            job="sbatch"
            script_path = opj(os.path.dirname(__file__),'HAB_G_fit_slurm')        
            # Arguments to pass to HAB_G_fit.py
            script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --n_jobs {n_jobs} {constraint} --prf_out {prf_out} --batch_id {batch_id} --batch_num {batch_num}"
            os.system(f'{job} {script_path} --job-name {prf_job_name} --output-dir {this_dir} --args "{script_args}"')
            # # sys.exit()