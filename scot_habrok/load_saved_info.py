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

derivatives_dir = '/scratch/p307263/pilot1/derivatives/'
fs_dir = opj(derivatives_dir, 'freesurfer')
code_dir = '/home4/p307263/programs/scot_habrok/scot_habrok'
log_dir = '/home4/p307263/programs/scot_habrok/logs'
default_ses = 'ses-1'

def load_yml_settings(hrf_version='old', sub=None):    
    yml_path = os.path.abspath(opj(code_dir, 's0_analysis_steps/s0_prf_analysis.yml'))
    with open(yml_path) as f:
        prf_settings = yaml.full_load(f)    
    
    for k in prf_settings.keys():
        if prf_settings[k]=='None':
            prf_settings[k] = None
    # UPDATE THE HRF
    if hrf_version=='old':
        prf_settings['hrf']['pars'] = [1, 1, 0]
    elif hrf_version=='new':
        prf_settings['hrf']['pars'] = [1, 4.6, 0]        
    elif hrf_version=='optimized':
        # DOESN'T matter...
        prf_settings['hrf']['pars'] = [1, 4.6, 0]                
    elif hrf_version=='optimal':
        prf_settings['hrf']['pars'] = [1, load_optimal_hrf(sub), 0]

    # if hrf_version=='old':
    #     yml_path = os.path.abspath(opj(code_dir, 's0_analysis_steps/s0_prf_analysis.yml'))        
    # elif hrf_version=='new':
    #     yml_path = os.path.abspath(opj(code_dir, 's0_analysis_steps/s0_prf_analysis_NEW_HRF.yml'))
    # elif hrf_version=='optimal':
    #     yml_path = os.path.abspath(opj(code_dir, 's0_analysis_steps/s0_prf_analysis_NEW_HRF.yml'))
        
    # # print(yml_path)
    # # yml_path = 'scot_habrok/s0_analysis_steps/s0_prf_analysis.yml'
    # with open(yml_path) as f:
    #     prf_settings = yaml.full_load(f)    

    # for k in prf_settings.keys():
    #     if prf_settings[k]=='None':
    #         prf_settings[k] = None
    # # UPDATE THE HRF
    # if hrf_version=='optimal':
    #     prf_settings['hrf']['pars'][1] = load_optimal_hrf(sub)

    return prf_settings

def load_optimal_hrf(sub):
    
    optimal_hrf_file = dag_find_file_in_folder(
        [sub, '.yml', 'OPTIMAL']   ,
        code_dir
    )
    print(f'Loading optimal HRF {optimal_hrf_file.split("/")[-1]}')

    with open(optimal_hrf_file) as f:
        optimal_hrf = yaml.full_load(f)    
    print(f'Loading optimal HRF {float(optimal_hrf["hrf_1"])}')
    return float(optimal_hrf['hrf_1'])


def get_scotoma_info():    
    scotoma_info = {}
    # Now get task specific info, and sort out the dodgy stuff...
    scotoma_info['AS0'] = {
        'scotoma_centre' : [],
        'scotoma_radius' : [],
    }

    # Is the distance to the screen the same?... 
    # need to convert from exp (I accidentally set screen distance to be 210, not 196)
    exptools_ssize = np.degrees(2 *np.arctan((39.3/2)/210))
    fitting_ssize = np.degrees(2 *np.arctan((39.3/2)/196))                
    conversion_factor = fitting_ssize / exptools_ssize
    # if (sub=="sub-01"): # & (ses=="ses-1"):        
    #     # need to convert from exp (I accidentally set screen distance to be 210, not 196)
    #     exptools_ssize = np.degrees(2 *np.arctan((39.3/2)/210))
    #     fitting_ssize = np.degrees(2 *np.arctan((39.3/2)/196))                
    #     conversion_factor = fitting_ssize / exptools_ssize
    # else:
    #     conversion_factor=1

    scotoma_info['AS1'] = {
        'scotoma_centre' : [0.8284* conversion_factor,0.8284* conversion_factor] ,
        'scotoma_radius' : 0.8284* conversion_factor,
    }
    scotoma_info['AS2'] = {
        'scotoma_centre' : [0* conversion_factor,0* conversion_factor] ,
        'scotoma_radius' : 2* conversion_factor,
    }    
    return scotoma_info
    

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
def load_roi(sub, roi, **kwargs):
    roi_idx = dag_load_roi(
        sub=sub, roi=roi, fs_dir=opj(derivatives_dir, 'freesurfer'), **kwargs)
    return roi_idx
    
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

    exclude = kwargs.get('exclude', ['batch'])
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

def load_prf_pickle_settings(prf_file):
    pkl_file = open(prf_file,'rb')
    pkl_data = pickle.load(pkl_file)
    pkl_file.close()     
    pkl_settings = pkl_data['settings']
    return pkl_settings


def get_design_matrix_npy(task_list):

    if not isinstance(task_list, list):
        task_list = [task_list]    
    dm_npy  = {}    
    for task in task_list:
        dm_path = dag_find_file_in_folder(['design', task], code_dir)        
        dm_npy[task] = scipy.io.loadmat(dm_path)['stim']
    return dm_npy
