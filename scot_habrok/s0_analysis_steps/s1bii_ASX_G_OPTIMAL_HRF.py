#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
from scot_habrok.load_saved_info import *
opj = os.path.join
prf_out = 'prf_NM_hrf4pt6_BL_full'
prf_dir = opj(derivatives_dir, prf_out)

sub_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_list = ['AS0',] # 'AS1', 'AS2']
model_list = ['gauss'] # ['gauss', 'norm']
batch_num = 20
roi_fit = 'v1custom'
constraint = '--nelder'
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    for model in model_list:
        for task in task_list:          
            job="python"
            script_path = opj(os.path.dirname(__file__),'HAB_G_fit_HRF_save_optimal.py')        
            # Arguments to pass to HAB_G_fit.py
            script_args = f"--sub {sub} --task {task} --model {model} --roi_fit {roi_fit} {constraint} --prf_out {prf_out} --batch_num {batch_num} --rsq_threshold 0.1"
            os.system(f'{job} {script_path} {script_args}')
            # sys.exit()