#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
from scot_habrok.load_saved_info import *
opj = os.path.join
prf_out = 'prf_HRFfit_NM_dt5'
prf_dir = opj(derivatives_dir, prf_out)

sub_list = ['sub-01','sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
task_list = ['AS0','AS1', 'AS2']
ses = 'ses-1'
detrend = 5
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)
    for task in task_list:          
        job="python"
        script_path = opj(os.path.dirname(__file__),'HAB_PSC.py')        
        # Arguments to pass to HAB_G_fit.py
        script_args = f"--sub {sub} --task {task} --prf_out {prf_out} --detrend {detrend}"
        os.system(f'{job} {script_path} {script_args}')
        # sys.exit()