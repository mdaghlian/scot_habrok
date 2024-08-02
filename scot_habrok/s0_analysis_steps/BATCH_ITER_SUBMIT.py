#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
opj = os.path.join
import sys
from scot_habrok.load_saved_info import *
from dag_prf_utils.utils import dag_hyphen_parse, dag_arg_checker

def main(argv):
    '''
    ---------------------------
    Auto open a subject surface with either numpy or pickle values

    Args:
        --prf_out       where to put everything
        --sub_list      subject list
        --task_list     what tasks to submit
        -m/--model      model to fit
        --constraint    what constraint to use
        --n_jobs        number of jobs
        --batch_num     how many batches
        --job_start     how many jobs in before starting
        --job_end       how many jobs in before ending...
        --ow            overwrite?
        ...
        --ses 
        --roi_fit

    '''
    prf_out = None
    sub_list = None
    task_list = None
    model = None
    constraint = None
    constraint_flag = ''
    hrf_version = 'new'
    hrf_fitting = False 
    batch_num = None
    n_jobs = None
    job_start = 0
    job_end = np.inf
    ow = False
    ow_flag = ''
    # 
    ses = 'ses-1'
    roi_fit = 'all'
    sl_args = '' # slurm arguments
    extra_kwargs = {}
    for i,arg in enumerate(argv):        
        if '--prf_out' in arg:                        
            prf_out = argv[i+1]            
        elif '--sub_list' in arg:
            sub_list = argv[i+1].split(',')
        elif '--task_list' in arg:
            task_list = argv[i+1].split(',')
        elif arg in ('-m', '--model'):
            model = argv[i+1]
        elif arg in ('--tc', '--bgfs', '--nelder'):
            constraint = arg.split('--')[-1]
            constraint_flag = arg
        elif '--n_jobs' in arg:
            n_jobs = int(argv[i+1])
        elif '--batch_num' in arg:
            batch_num = int(argv[i+1])
        elif '--job_start' in arg:
            job_start = int(argv[i+1])
        elif '--job_end' in arg:
            job_end = int(argv[i+1])            
        elif '--ses' in arg:
            ses = dag_hyphen_parse('ses', argv[i+1])
        elif '--roi_fit' in arg:
            roi_fit = argv[i+1]
        elif '--hrf_version' in arg:
            hrf_version = argv[i+1]   
        elif '--hrf_fitting'==arg:
            hrf_fitting = True
        elif arg in ("--ow", "--overwrite"):
            ow = True            
            ow_flag = '--ow'
        elif arg in ('-h', '--help'):
            print(main.__doc__)
            sys.exit()
        elif arg in ('--time', '--nodes', '--ntasks-per-node'):
            sl_args += f' {arg} {argv[i+1]}'
        elif '--' in arg:
            this_kwarg = arg.replace('--', '')
            this_kwarg_value = dag_arg_checker(argv, i+1)
            extra_kwargs[this_kwarg] = this_kwarg_value
            print(f'Unknown arg: {arg}')

    # Sort out paths
    prf_dir = opj(derivatives_dir, prf_out)
    prf_log_dir = opj(log_dir, prf_out)
    i = 0
    for sub in sub_list:
        p_dir = opj(prf_dir, sub, ses)
        l_dir = opj(prf_log_dir, sub, ses)
        # Make specific sub folder for the batch
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)
        if not os.path.exists(l_dir):
            os.makedirs(l_dir)
        for task in task_list:        
            for batch_id in np.arange(1,batch_num+1):
                hrf_fitting_str = ''
                if hrf_fitting:
                    hrf_fitting_str = '_HRFFIT_'
                prf_job_name = f'{sub}-{model}-{task}{hrf_version}{hrf_fitting_str}-iter-{batch_id:03}-of-{batch_num:03}'
                output_file = os.path.abspath(opj(l_dir, f'{prf_job_name}_OUT.txt'))
                error_file = os.path.abspath(opj(l_dir, f'{prf_job_name}_ERR.txt'))
                slurm_args = f'--output {output_file} --error {error_file} --job-name {prf_job_name}' + \
                    sl_args

                i += 1
                if i<job_start:
                    print(f'skipping job number {sub} {task} {batch_id}')
                    continue
                if i>job_end:
                    print(f'reached job number {i}... stopping')
                    return
                
                job="sbatch"
                slurm_path = opj(os.path.dirname(__file__),'HAB_SLURM_GENERIC')        
                if model=='gauss':
                    script_path = opj(os.path.dirname(__file__),'HAB_G_fit.py')        
                elif model in ('norm', 'css', 'dog'):
                    script_path = opj(os.path.dirname(__file__),'HAB_N_fit.py')        
                
                if hrf_fitting:
                    script_path = opj(os.path.dirname(__file__),'HAB_G_fit_HRF.py')
                    model = 'gauss'        
                batch_str = f'_batch-{batch_id:03}-of-{batch_num:03}'
                iter_check = dag_find_file_in_folder(
                    [task, roi_fit, model, 'iter', f'constr-{constraint}', f'hrf-{hrf_version}', batch_str, '.pkl'], 
                    p_dir, 
                    return_msg=None)
                # continue
                # print([task, roi_fit, model, 'iter', f'constr-{constraint}', batch_str, '.pkl'])
                # print(p_dir)
                # print(f'OW = {ow}')
                if (iter_check is not None) & (not ow):
                    print(f'Already done : {iter_check.split("/")[-1]}')
                    i -= 1
                    continue
                # Arguments to pass to fitters
                script_args = f"--sub {sub} --task {task} --model {model} " +\
                    f"--roi_fit {roi_fit} --n_jobs {n_jobs} {constraint_flag} --prf_out {prf_out} " +\
                    f"--batch_id {batch_id} --batch_num {batch_num} --hrf_version {hrf_version} " +\
                    f"{ow_flag} "
                # print(f'python {script_path} {script_args}')
                # Run locally 
                # print(f'python {script_path} {script_args}')
                # os.system(f'python {script_path} {script_args}')

                # Submit!
                os.system(f'{job} {slurm_args} {slurm_path} --script_path {script_path} --args "{script_args}"')
                # sys.exit()
    print(i)    


if __name__ == "__main__":
    main(sys.argv[1:])    


