import numpy as np
import scipy.io
import yaml
import pickle
import os
opj = os.path.join
import pandas as pd
try:
    from prfpy_csenf.stimulus import PRFStimulus2D
except:
    from prfpy.stimulus import PRFStimulus2D

from dag_prf_utils.utils import *
from dag_prf_utils.prfpy_functions import set_tc_shape
derivatives_dir = '/scratch/p307263/pilot1/derivatives/prf_habrok'
code_dir = '/home4/p307263/programs/scot_habrok/scot_habrok'
default_ses = 'ses-1'

def load_yml_settings():    
    # yml_path = os.path.abspath(opj(code_dir, 's0_analysis_steps/s0_prf_analysis.yml'))
    # print(yml_path)
    yml_path = 'scot_habrok/s0_analysis_steps/s0_prf_analysis.yml'
    with open(yml_path) as f:
        prf_settings = yaml.full_load(f)    
    return prf_settings

def load_data_tc(sub, task_list, ses=default_ses, look_in=None, do_demo=False, n_timepts=None):
    '''
    Loads real data
    '''
    look_in = opj(derivatives_dir, look_in)
    if isinstance(task_list, str):
        task_list = [task_list]
    data_tc  = {}
    this_dir = opj(look_in, sub, ses)
    for task in task_list:
        try:
            data_tc_path = dag_find_file_in_folder([sub, ses, dag_hyphen_parse('task', task), 'hemi-LR', '.npy'], this_dir, exclude=['correlation', 'mean_epi'])
        except:
            data_tc_path = dag_find_file_in_folder([sub, ses, dag_hyphen_parse('task', task), 'hemi-lr', '.npy'], this_dir, exclude=['correlation', 'mean_epi'])
        data_tc[task] = set_tc_shape(np.load(data_tc_path), n_timepts=n_timepts)
        if do_demo:
            data_tc[task] = data_tc[task][0:100,:]
    return data_tc

def load_data_prf(sub, task_list, model_list, var_to_load='pars', roi_fit='all', fit_stage='iter', ses=default_ses, look_in=None, **kwargs):
    '''
    Load PRF model fits on * DATA *  (pkl file)
    output a dict 
        prf_vars[task][model]
    Default loads 'pars' (prf params)
    Can also specify settings or preds 
    '''
    look_in = opj(derivatives_dir, look_in)
    include = kwargs.get('include', [])
    if isinstance(include, str):
        include = [include]

    exclude = kwargs.get('exclude', None)
    if isinstance(exclude, str):
        exclude = [exclude]
            
    if isinstance(task_list, str):
        task_list = [task_list]
    if isinstance(model_list, str):
        model_list = [model_list]

    
    prf_vars  = {}
    this_dir = opj(look_in, sub, ses)
    for task in task_list:
        prf_vars[task] = {}
        for model in model_list:
            this_include = include + [sub, dag_hyphen_parse('task', task), model, roi_fit, fit_stage, '.pkl']
            prf_vars_path = dag_find_file_in_folder(this_include, this_dir, exclude=exclude)            
            print(prf_vars_path)
            pkl_file = open(prf_vars_path,'rb')
            pkl_data = pickle.load(pkl_file)
            pkl_file.close()     
            if 'pred' in var_to_load:
                prf_vars[task][model] = set_tc_shape(pkl_data[var_to_load])
            else:
                prf_vars[task][model] = pkl_data[var_to_load]

    return prf_vars  

def load_prf_pickle_pars(prf_file):
    pkl_file = open(prf_file,'rb')
    pkl_data = pickle.load(pkl_file)
    pkl_file.close()     
    pkl_pars = pkl_data['pars']
    return pkl_pars



def get_design_matrix_npy(task_list):

    if not isinstance(task_list, list):
        task_list = [task_list]    
    dm_npy  = {}    
    for task in task_list:
        dm_path = dag_find_file_in_folder(['design', task], code_dir)        
        dm_npy[task] = scipy.io.loadmat(dm_path)['stim']
    return dm_npy
