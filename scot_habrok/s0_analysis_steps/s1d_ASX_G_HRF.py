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
sl_time = '5:00:00'

# Where is it going? What HRF version is being used
prf_out = 'prf_NM_hrf4pt6_BL_full'
hrf_version = 'new'
n_jobs = 64
batch_num = 20
roi_fit = 'v1custom'
constraint = '--nelder'
ses = 'ses-1'
model = 'gauss'
ow = False

prf_dir = opj(derivatives_dir, prf_out)
prf_log_dir = opj(log_dir, prf_out)
sub_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
task_list = [ 'AS0', 'AS1', 'AS2']
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    this_dir = opj(prf_dir, sub, ses)
    this_log_dir = opj(prf_log_dir, sub, ses)
    if not os.path.exists(this_log_dir):
        os.makedirs(this_log_dir)
    for task in task_list:    
        prf_job_name = f'{sub}-gauss-HRF'
        # done_check = dag_find_file_in_folder(
        #     [model, roi_fit, task, 'grid', '.pkl'],
        #     path=this_dir,
        #     return_msg=None,
        # )
        # if (done_check is not None) & (not ow):
        #     print(f'Already done {done_check}')
        #     continue


        output_file = os.path.abspath(opj(this_log_dir, f'{prf_job_name}_OUT.txt'))
        error_file = os.path.abspath(opj(this_log_dir, f'{prf_job_name}_ERR.txt'))
        # slurm_args = f'--output {output_file} --error {error_file} --job-name {prf_job_name}'
        for batch_id in np.arange(1,batch_num+1):

            job="python"
            # job="sbatch"
            script_path = opj(os.path.dirname(__file__),'HAB_G_fit_HRF.py')        
            # Arguments to pass to HAB_G_fit.py
            script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --n_jobs {n_jobs} " + \
                f"{constraint} --prf_out {prf_out} --batch_num {batch_num} --batch_id {batch_id} " +\
                f"--hrf_version {hrf_version}"
            # os.system(f'bash {script_path} --args "{script_args}"')
            os.system(f'python {script_path} {script_args}')
            sys.exit()