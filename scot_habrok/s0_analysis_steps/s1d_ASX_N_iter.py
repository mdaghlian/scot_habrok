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
prf_log_dir = opj(log_dir, prf_out)

model = 'norm'
sub_list = ['sub-01', ] # 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
sub_list = ','.join(sub_list)
task_list = ['AS1', 'AS2']
task_list = ','.join(task_list)
n_jobs = 64
batch_num = 20
roi_fit = 'all'
constraint = '--tc'
ses = 'ses-1'
script_path = opj(os.path.dirname(__file__),'BATCH_ITER_SUBMIT.py')        

os.system(
    f'python {script_path} --prf_out {prf_out} --sub_list {sub_list} --task_list {task_list} ' + \
    f'--model {model} {constraint} --n_jobs {n_jobs} --batch_num {batch_num}'
)