#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
opj = os.path.join
from dag_prf_utils.utils import dag_get_cores_used
import time


derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives'
prf_out = 'prf_half'
prf_dir = opj(derivatives_dir, prf_out)

sub_list = ['sub-04', 'sub-05', 'sub-06']
roi_fit = 'all'
constraint = '--tc'
nr_jobs = 15

task_list = ['AS0_run-1', 'AS0_run-2']
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for task in task_list:
        prf_job_name = f'G-{task}-{sub}-{roi_fit}'            
        job=f"qsub -q long.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
        # job='python'
        script_path = opj(os.path.dirname(__file__),'s1b_G_fit.py')
        script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --prf_out {prf_out}"
        n_cores = dag_get_cores_used()
        while n_cores>=60:
            # pause for 
            print('sleep')
            time.sleep(50)
            n_cores = dag_get_cores_used()        

        os.system(f'{job} {script_path} {script_args}')
        # sys.exit()