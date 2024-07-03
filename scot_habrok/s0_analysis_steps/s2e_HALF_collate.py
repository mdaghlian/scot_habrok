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

model_list = ['norm'] #  ['gauss', 'norm']
sub_list = ['sub-01'] # ,'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_list = ['AS0_run-1', 'AS0_run-2', 'AS1_run-1', 'AS1_run-2', 'AS2_run-1', 'AS2_run-2']
batch_num = 20
roi_fit = 'all'
constraint = '--tc'
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for model in model_list:
        for task in task_list:          
            job="python"
            script_path = opj(os.path.dirname(__file__),'HAB_collate.py')        
            script_args = f"--sub {sub} --task {task} --model {model} --roi_fit {roi_fit} {constraint} --prf_out {prf_out} --batch_num {batch_num}"
            os.system(f'{job} {script_path} {script_args}')
            # sys.exit()
