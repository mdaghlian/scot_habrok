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

Take a fit from the time series and then fit the dispersion for the HRF. 
Look only in roi 

Args:
    -s (--sub=)         e.g., 01
    -t (--task=)        task (AS0, AS1, AS2)
    --batch_id          id giving the batch to run
    --batch_num         how many batches in total
    --grid_only         only run the grid
    --n_jobs           number of jobs
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
    hrf_version = 'optimized'
    
    # Specify
    sub = None
    task = None
    roi_fit = 'custom'
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
        elif '--batch_id' in arg:
            batch_id = int(argv[i+1])
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
    
    roi_idx = load_roi(sub, roi_fit, combine_matches=True)
    print(f'For ROI {roi_fit} % of vx being fit = {roi_idx.mean()*100:.3f}')
    if batch_num is not None:        
        batch_str = f'_batch-{batch_id:03}-of-{batch_num:03}'
    else:
        batch_str = ''

    out = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{dag_hyphen_parse('hrf', hrf_version)}_{task}-fits{batch_str}_HRF-refit"    
    out_no_batch_str = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{task}-fits_HRF-refit"    
    
    # TO FIND THE MATCHING NOT FIT HRF FILE
    out_hrf_not_fit = [
        sub, dag_hyphen_parse('model', model), task, f'fits{batch_str}' 
        # Ignoring ROI
    ]

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD SETTINGS
    # load basic settings from the yml file
    prf_settings = load_yml_settings(hrf_version=hrf_version, sub=sub)
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
    
    # Split into batches?    
    if batch_num is not None:
        ts_data, ts_idx = dag_split_mat_with_idx(
            ts_data, batch_num=batch_num, batch_id=batch_id-1, # NOTE MINUS 1
            split_method='distributed',
        )
        batch_roi_idx = roi_idx[ts_idx]
        print(f'ts shape = {ts_data.shape}')
        print(ts_idx)
        print(batch_roi_idx)
        print(f'Running {batch_str}')
        print(f'In this batch there are {len(batch_roi_idx)} vx')
        print(f'Of wich {batch_roi_idx.sum()} are included in roi {roi_fit}')
        # save  
        batch_idx_file = opj(output_dir, f'{out}_batch-idx.npy')
        np.save(batch_idx_file, ts_idx)
    else:
        batch_roi_idx = roi_idx #np.ones(ts_data.shape[0])

    total_num_vx = batch_roi_idx.shape[0]
    fitting_num_vx = batch_roi_idx.sum()
    ts_data = ts_data[batch_roi_idx]
    print(f'Fitting {roi_fit} batch {batch_id} out of {batch_num} vx in batch={total_num_vx}, vx fitting = {fitting_num_vx} ')
    print(ts_data.shape)
    print(f'RSQ THRESHOLD {prf_settings["rsq_threshold"]}')

    # Look for the parameters
    print([out, 'gauss', 'iter', constraints])
    iter_check = dag_find_file_in_folder([out, 'gauss', 'iter', dag_hyphen_parse('constr', constraints)], output_dir, return_msg=None)
    if (iter_check is not None) and (not overwrite):
        print(f'Already done {iter_check}')
        sys.exit()        

    # NOW find the relevant hrf_not_fit file
    iter_hrf_not_fit = dag_find_file_in_folder(
        [*out_hrf_not_fit, 'gauss', 'iter', dag_hyphen_parse('constr', constraints)], 
        output_dir, 
        exclude=['HRF-refit', 'HRF-specific'],
        )
    print(out_hrf_not_fit)
    # print(iter_hrf_not_fit)
    # Now load it 
    iter_hrf_not_fit_pars = load_prf_pickle_pars(iter_hrf_not_fit) # 

    # What if there are no vx to fit??? 
    do_fitting = True
    if fitting_num_vx==0:
        print('Nothing to fit here!!')
        do_fitting = False
        # Fill it all with zeros
        iter_pars_to_save = np.zeros_like(iter_hrf_not_fit_pars)
        i_start_time=0
        i_end_time=0

    # ************************************************************************

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   LOAD DESIGN MATRIX   
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
    # ************************************************************************
    
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CREATE GAUSSIAN MODEL & fitter   
    gg = Iso2DGaussianModel(
        stimulus=prf_stim,                                  # The stimulus we made earlier
        hrf=prf_settings['hrf']['pars'],                    # These are the parameters for the HRF that we normally use at Spinoza (with 7T data). (we can fit it, this will be done later...)
        normalize_RFs=prf_settings['normalize_RFs'],        # Normalize the volume of the RF (so that RFs w/ different sizes have the same volume. Generally not needed, as this can be solved using the beta values i.e.,amplitude)
        )
    gf = Iso2DGaussianFitter(
        data=ts_data,             # time series
        model=gg,                       # model (see above)
        n_jobs=prf_settings['n_jobs'], # number of jobs to use in parallelization 
        )    
    # ************************************************************************
    if do_fitting:
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DO ITERATIVE FIT
        gf.gridsearch_params = iter_hrf_not_fit_pars[batch_roi_idx,:]        

        # Stuff to print:         
        vx_gt_rsq_th = gf.gridsearch_params[:,-1]>prf_settings['rsq_threshold']
        nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
        mean_vx_gt_rsq_th = np.mean(gf.gridsearch_params[vx_gt_rsq_th,-1])
        print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')    
        print(f'Now to refit the HRF')
        
        gauss_bounds = [
            (-1.5*max_eccentricity, 1.5*max_eccentricity),          # x bound
            (-1.5*max_eccentricity, 1.5*max_eccentricity),          # y bound
            (1e-1, max_eccentricity*3),                             # prf size bounds
            (prf_settings['prf_ampl'][0],prf_settings['prf_ampl'][1]),      # prf amplitude
            (prf_settings['bold_bsl'][0],prf_settings['bold_bsl'][1]),      # bold baseline (fixed)
            (prf_settings['hrf']['deriv_bound'][0], prf_settings['hrf']['deriv_bound'][1]), # hrf_1 bound
            (prf_settings['hrf']['disp_bound'][0],  prf_settings['hrf']['disp_bound'][1]), # hrf_2 bound
        ]
        print(gauss_bounds)
        print(f'Now fixing all the gauss parameters apart from the amplitude and HRF')
        model_idx = prfpy_params_dict()['gauss']
        # Which parameters to fix? 
        fix_param_dict = {}
        for p in model_idx.keys():
            if ('hrf' in p) or (p=='amp_1'):
                continue
            else:
                fix_param_dict[p] = gf.gridsearch_params[:, model_idx[p]]
        
        gauss_bounds = make_vx_wise_bounds(
            fitting_num_vx, 
            gauss_bounds, 
            model='gauss', 
            fix_param_dict = fix_param_dict
            )   
        print(gauss_bounds)
        # Constraints determines which scipy fitter is used
        # -> can also be used to make certain parameters interdependent (e.g. size depening on eccentricity... not normally done)
        if prf_settings['constraints']=='tc':
            g_constraints = []   # uses trust-constraint (slower, but moves further from grid
            minimize_args = {}
        elif prf_settings['constraints']=='bgfs':
            g_constraints = None # uses l-BFGS (which is faster)
            minimize_args = {}
        elif prf_settings['constraints']=='nelder':
            g_constraints = []
            minimize_args = dict(
                method='nelder-mead',            
                options=dict(disp=False),
                constraints=[],
                tol=float(prf_settings['ftol']),
                )
            
        i_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        print(f'Starting iter {i_start_time}, constraints = {g_constraints}')
        start = time.time()

        gf.iterative_fit(
            rsq_threshold=prf_settings['rsq_threshold'],    # Minimum variance explained. Puts a lower bound on the quality of PRF fits. Any fits worse than this are thrown away...     
            verbose=True,
            bounds=gauss_bounds,       # Bounds (on parameters)
            constraints=g_constraints, # Constraints
            xtol=float(prf_settings['xtol']),     # float, passed to fitting routine numerical tolerance on x
            ftol=float(prf_settings['ftol']),     # float, passed to fitting routine numerical tolerance on function
            minimize_args=minimize_args,
            )

        # Fiter for nans
        gf.iterative_search_params = dag_filter_for_nans(gf.iterative_search_params)    
        i_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        print(f'End iter {i_end_time}')           
        elapsed = (time.time() - start)
        print(f'Finished iter {i_end_time}')
        print(f'Took {timedelta(seconds=elapsed)}')
        vx_gt_rsq_th = gf.iterative_search_params[:,-1]>prf_settings['rsq_threshold']
        nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
        mean_vx_gt_rsq_th = np.mean(gf.iterative_search_params[vx_gt_rsq_th,-1]) 
        print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')
        
        iter_pars_to_save = np.zeros_like(iter_hrf_not_fit_pars)
        iter_pars_to_save[batch_roi_idx,:] = gf.iterative_search_params   
        # *************************************************************


    # Save everything as a pickle...
    iter_pkl_file = opj(output_dir, f'{out}_stage-iter_constr-{constraints}_desc-prf_params.pkl')
    iter_dict = {}
    iter_dict['pars'] = iter_pars_to_save
    iter_dict['settings'] = prf_settings
    iter_dict['start_time'] = i_start_time
    iter_dict['end_time'] = i_end_time
    iter_dict['prfpy_model'] = gg

    from figure_finder.utils import get_running_path, get_running_code_string
    iter_dict['running_code_string'] = get_running_code_string(get_running_path())

    # Dump everything!!! into pickle
    f = open(iter_pkl_file, "wb")
    pickle.dump(iter_dict, f)
    f.close()

    # Dump running code 
    iter_pkl_file = opj(output_dir, f'{out_no_batch_str}_stage-iter_constr-{constraints}_desc-prf_params.pkl')
    run_code_file = iter_pkl_file.replace('prf_params.pkl', 'running_code.py')
    with open(run_code_file, 'w') as f:
        f.write(iter_dict['running_code_string'])
    
    # Also dump the settings as a separate yaml file for ease of reading 
    settings_file = iter_pkl_file.replace('prf_params.pkl', 'settings.yml')
    with open(settings_file, 'w') as f:
        yaml.dump(prf_settings, f)


    print('DONE!!!')


if __name__ == "__main__":
    main(sys.argv[1:])
