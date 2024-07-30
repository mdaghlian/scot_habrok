#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import sys
import os
opj = os.path.join
import numpy as np
import yaml
import pickle
from datetime import datetime, timedelta
import time

try:
    from prfpy.stimulus import PRFStimulus2D
    from prfpy.model import Iso2DGaussianModel
    from prfpy.fit import Iso2DGaussianFitter
except:
    from prfpy_csenf.stimulus import PRFStimulus2D
    from prfpy_csenf.model import Iso2DGaussianModel
    from prfpy_csenf.fit import Iso2DGaussianFitter

from dag_prf_utils.utils import *
from dag_prf_utils.prfpy_functions import *

from scot_habrok.load_saved_info import *

def main(argv):

    """
---------------------------------------------------------------------------------------------------

Load the fit HRF params -> get the rsq weighted mean HRF
Save it into hrf_optimised
Test it on old, HRF unfit...

Look only in roi 

Args:
    -s (--sub=)         e.g., 01
    -t (--task=)        task (AS0, AS1, AS2)
    --roi_fit           Which ROI optimized for (and to optimize for)
    --batch_num         how many batches in total
    --verbose
    --tc                
    --bgfs
    --ow               overwrite    
    

Example:


---------------------------------------------------------------------------------------------------
    """
    print('\n\n\n\n')

    # default
    ses = 'ses-1'
    model = 'gauss'
    verbose = True
    cut_vols = 5
    n_timepts = 225 - cut_vols
    hrf_version = 'new'
    max_ecc = 5
    
    # Specify
    sub = None
    task = None
    roi_fit = 'v1custom'
    constraints = None
    n_jobs = None
    prf_out = 'prf'    
    overwrite = False
    rsq_threshold = None
    ow_prf_settings = {} # overwrite prf settings from the yml file with these settings
    batch_id = None
    batch_num = None

    for i,arg in enumerate(argv):
        if arg in ('-s', '--sub'):
            sub = dag_hyphen_parse('sub', argv[i+1])
        elif arg in ('--ses'):
            ses = dag_hyphen_parse('ses', argv[i+1])            
        elif arg in ('-t', '--task'):
            task = dag_hyphen_parse('task', argv[i+1])
        elif '--prf_out' in arg:
            prf_out = argv[i+1]   
        elif '--batch_num' in arg:
            batch_num = int(argv[i+1])            
        elif arg in ("-r", "--roi_fit"):
            roi_fit = argv[i+1]
        elif arg in ("--n_jobs"):
            n_jobs = int(argv[i+1])  
        elif arg in ("--tc", "--bgfs", "--nelder"):
            constraints = arg.split('--')[-1]
        elif arg in ("--rsq_threshold",):
            rsq_threshold = float(argv[i+1])                        
        elif arg in ("--hrf_version",):
            hrf_version = argv[i+1]
        elif arg in ("--ow", "--overwrite"):
            overwrite = True
        elif arg in ('-h', '--help'):
            print(main.__doc__)
            sys.exit(2)
        elif '--' in arg:
            ow_prf_settings[arg.split('--')[-1]] = dag_arg_checker(argv[i+1])    

    # Where to save everything
    prf_dir = opj(derivatives_dir,  prf_out)    
    output_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    # Load relevant roi file 
    out = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{task}-fits_HRF-OPTIMAL"    
    
    # TO FIND THE MATCHING FIT FILEs
    out_hrf_fit_filt = [
        sub, 'COLLATED', task, 
        dag_hyphen_parse('roi', roi_fit),
        'hrf-optimized',
        'iter', dag_hyphen_parse('constr', constraints),
        '.pkl',
    ]
    out_hrf_fit = dag_find_file_in_folder(
        out_hrf_fit_filt, 
        output_dir, 
        exclude='batch',
        return_msg=None)
    print(out_hrf_fit)

    # TO FIND THE MATCHING NOT FIT HRF FILE
    out_NOT_hrf_fit_filt = [
        sub, dag_hyphen_parse('model', model), task, 
        'roi-all',
        'iter', dag_hyphen_parse('constr', constraints),    
        '.pkl',   
    ]    
    out_NOT_hrf_fit = dag_find_file_in_folder(
        out_NOT_hrf_fit_filt, 
        output_dir, 
        exclude=['hrf-optimal', 'batch'],
        return_msg=None)
    print(out_NOT_hrf_fit)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD SETTINGS
    # load basic settings from the yml file
    prf_settings = load_yml_settings('new')
    dm_task = task +''
    dm_task = dm_task.split('_run')[0] # 
    dm_task = dm_task.split('_fold')[0] 
    # Add important info to settings
    prf_settings['sub'] = sub
    prf_settings['task'] = task
    prf_settings['model'] = model
    prf_settings['roi_fit'] = roi_fit
    prf_settings['n_jobs'] = n_jobs
    prf_settings['constraints'] = constraints
    prf_settings['ses'] = ses
    prf_settings['task'] = task
    prf_settings['fit_hrf'] = True
    prf_settings['verbose'] = verbose
    prf_settings['prf_out'] = out 
    prf_settings['prf_dir'] = prf_dir
    prf_settings['cut_vols'] = cut_vols
    prf_settings['n_timepts'] = n_timepts    
    if rsq_threshold!=None:        
        prf_settings['rsq_threshold'] = rsq_threshold
    else:
        rsq_threshold = prf_settings['rsq_threshold']
    if len(ow_prf_settings)>0:
        for key in ow_prf_settings.keys():
            prf_settings[key] = ow_prf_settings[key]
            print(f'Overwriting {key} with {ow_prf_settings[key]}')

    # ****************************************************
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD TIME SERIES
    ts_data = load_data_tc(
        sub=sub, 
        ses=ses, 
        task_list=task, 
        look_in=prf_dir, 
        n_timepts=n_timepts
        )[task]
    
    roi_idx = load_roi(sub, roi_fit)
    total_num_vx = ts_data.shape[0]
    vx_in_roi = roi_idx.sum()
    print(f'Num vs in total = {total_num_vx}')
    print(f'Num vx in ROI = {vx_in_roi}')


    # Find the "optimal HRF"
    print(f'Finding the hrf_1 rsq weighted mean in {roi_fit}')
    # load parameters
    hrf_fit_pars = load_prf_pickle_pars(out_hrf_fit) # 
    # inside roi mask, rsq > 0, ecc < max_ecc
    ecc = np.sqrt(hrf_fit_pars[:,0]**2 + hrf_fit_pars[:,1]**2) < max_ecc
    rsq = hrf_fit_pars[:,-1] > 0
    full_mask = ecc & rsq & roi_idx
    print(f'Number of vs in full mask = {full_mask.sum()}')

    from dag_prf_utils.utils import dag_weighted_mean, dag_get_rsq
    rsq_for_weight = hrf_fit_pars[full_mask,-1]
    hrf_1_values = hrf_fit_pars[full_mask,-3]
    w_hrf_1 = dag_weighted_mean(
        w=rsq_for_weight,
        x=hrf_1_values,
    )
    print('******************')
    print(f'{sub} {task} {w_hrf_1:.3f}')
    # save to 
    # Also dump the settings as a separate yaml file for ease of reading 
    optimal_file = opj(code_dir, f'{out}.yml')
    with open(optimal_file, 'w') as f:
        yaml.dump({'hrf_1': f'{w_hrf_1:.3f}'}, f)    

    # NOW compare, does it improve? 
    # NOT hrf fit pars
    NOT_hrf_fit_pars = load_prf_pickle_pars(out_NOT_hrf_fit) # 
    ecc_mask = np.sqrt(NOT_hrf_fit_pars[:,0]**2 + NOT_hrf_fit_pars[:,1]**2) < max_ecc
    rsq_mask = NOT_hrf_fit_pars[:,-1]>rsq_threshold
    rsq_ecc_mask = rsq_mask & ecc_mask
    print(f'OLD hrf value ={prf_settings["hrf"]["pars"]}')
    print(f'OLD n voxels > {rsq_threshold}, = {rsq_ecc_mask.sum()}')
    m_rsq = np.mean(NOT_hrf_fit_pars[rsq_ecc_mask,-1])
    m_rsq_roi = np.mean(NOT_hrf_fit_pars[rsq_ecc_mask&roi_idx,-1])
    m_rsq_NOT_roi = np.mean(NOT_hrf_fit_pars[rsq_ecc_mask&~roi_idx,-1])
    print(f'Mean rsq for ALL vx > {rsq_threshold} = {m_rsq:.3f} ')
    print(f'Mean rsq for {roi_fit} vx > {rsq_threshold} = {m_rsq_roi:.3f} ')
    print(f'Mean rsq for NOT {roi_fit} vx > {rsq_threshold} = {m_rsq_NOT_roi:.3f} ')

    print('NOW running with "optimal" HRF')

    design_matrix = get_design_matrix_npy([dm_task])[dm_task]         
    design_matrix = design_matrix[:,:,cut_vols:]
    assert design_matrix.shape[-1]==ts_data.shape[-1]
    prf_stim = PRFStimulus2D(
        screen_size_cm=prf_settings['screen_size_cm'],          # Distance of screen to eye
        screen_distance_cm=prf_settings['screen_distance_cm'],  # height of the screen (i.e., the diameter of the stimulated region)
        design_matrix=design_matrix,                            # dm (npix x npix x time_points)
        TR=prf_settings['TR'],                                  # TR
        )   
    max_eccentricity = prf_stim.screen_size_degrees/2 # It doesn't make sense to look for PRFs which are outside the stimulated region 
    
    
    new_hrf_pars = prf_settings['hrf']['pars']
    new_hrf_pars[1] = w_hrf_1
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CREATE GAUSSIAN MODEL & fitter   
    gg = Iso2DGaussianModel(
        stimulus=prf_stim,                                  # The stimulus we made earlier
        hrf=new_hrf_pars,
        normalize_RFs=prf_settings['normalize_RFs'],        # Normalize the volume of the RF (so that RFs w/ different sizes have the same volume. Generally not needed, as this can be solved using the beta values i.e.,amplitude)
        )
    pars_for_preds = NOT_hrf_fit_pars[rsq_ecc_mask,:]
    pars_for_preds[:,-3] = w_hrf_1
    print(pars_for_preds[0,:])
    new_preds = gg.return_prediction(*list(pars_for_preds[:,:-1].T))
    print(new_preds.shape)
    ts_target = ts_data[rsq_ecc_mask,:]
    new_rsq = np.zeros_like(rsq_ecc_mask, dtype=float)
    new_rsq[rsq_ecc_mask] = dag_get_rsq(
        ts_target, new_preds,
    )
    m_rsq = np.mean(new_rsq[rsq_ecc_mask])
    m_rsq_roi = np.mean(new_rsq[rsq_ecc_mask&roi_idx])
    m_rsq_NOT_roi = np.mean(new_rsq[rsq_ecc_mask&~roi_idx])
    print(f'Mean rsq for ALL vx > {rsq_threshold} = {m_rsq:.3f} ')
    print(f'Mean rsq for {roi_fit} vx > {rsq_threshold} = {m_rsq_roi:.3f} ')
    print(f'Mean rsq for NOT {roi_fit} vx > {rsq_threshold} = {m_rsq_NOT_roi:.3f} ')
    print('DONE!!!')


if __name__ == "__main__":
    main(sys.argv[1:])
