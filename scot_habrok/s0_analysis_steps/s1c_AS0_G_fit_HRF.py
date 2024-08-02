#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
from scot_habrok.load_saved_info import *
opj = os.path.join


# SLURM ARGUMENTS (partition & profile included in the task)
sl_nodes = '1'
sl_task_per_node = '30'
sl_time = '0:15:00'

# Where is it going? What HRF version is being used
prf_out = 'prf_HRFfit_NM_dt5'
hrf_version = 'optimized' # ONLY OPTIMIZING THE HRF
n_jobs = 64
batch_num = 20
roi_fit = 'custom'
constraint = '--nelder'
ses = 'ses-1'
model = 'gauss'
ow = False
ow_flag = ''

sub_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
sub_list = ','.join(sub_list)
task_list = 'AS0'

script_path = opj(os.path.dirname(__file__),'BATCH_ITER_SUBMIT.py')        
sl_args = f' --time {sl_time} --nodes {sl_nodes} --ntasks-per-node {sl_task_per_node}'

os.system(
    f'python {script_path} --prf_out {prf_out} --sub_list {sub_list} --task_list {task_list} ' + \
    f'--model {model} {constraint} --n_jobs {n_jobs} --batch_num {batch_num} ' + \
    f'--roi_fit {roi_fit} --hrf_version {hrf_version}  --hrf_fitting {ow_flag} ' +\
    sl_args
)