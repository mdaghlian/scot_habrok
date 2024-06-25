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

Fit the time series using the gaussian 

Args:
    -s (--sub=)         e.g., 01
    -t (--task=)        task (AS0, AS1, AS2)
    -m (--model=)       
    --batch_num 
    --roi_fit
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
    fit_hrf = False
    verbose = True
    
    # Specify
    sub = None
    task = None
    roi_fit = 'all'
    constraints = None
    prf_out = 'prf'    
    overwrite = False
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
        elif arg in ("--tc"):
            constraints = "tc"
        elif arg in ("--bgfs"):
            constraints = "bgfs"
        elif arg in ("--ow" or "--overwrite"):
            overwrite = True
        elif arg in ('-h', '--help'):
            print(main.__doc__)
            sys.exit(2)

    # Where to save everything
    prf_dir = opj(derivatives_dir,  prf_out)    
    output_dir = opj(prf_dir, sub, ses)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    out = f"{sub}_{model}_{roi_fit}_{task}-fits_COLLATED"    
    
    batch_idx = []
    batch_pars = []
    for ib in np.arange(1,batch_num+1):
        # Find the pickle file
        batch_pkl_file = dag_find_file_in_folder(
            [sub, model, task, roi_fit, 'iter', constraints, '.pkl', f'batch-{ib:03}-of-{batch_num:03}'],  output_dir, 
            return_msg=None, 
            )
        if ib==1:
            # LOAD ALL THE SETTINGS + PARS... we will put them in later
            pkl_file = open(batch_pkl_file,'rb')
            pkl_data = pickle.load(pkl_file)
            pkl_file.close()     
        batch_pars.append(
            load_prf_pickle_pars(batch_pkl_file)
        )
        
        # Also load the index            
        batch_idx_file = dag_find_file_in_folder(
            [sub, model, task, roi_fit, 'batch-idx.npy', f'batch-{ib:03}-of-{batch_num:03}'],  output_dir, 
            return_msg=None, 
            )
        batch_idx.append(np.load(batch_idx_file))
    
    # Find the total number of voxels and params
    total_n_params = batch_pars[0].shape[-1]
    total_n_vox = np.concatenate(batch_idx).max() + 1 # (index)
    print(f'total n_vox = {total_n_vox}')
    pkl_data['pars'] = np.zeros((total_n_vox, total_n_params))
    for idx,pars in zip(batch_idx, batch_pars):
        pkl_data['pars'][idx,:] = pars
    iter_pkl_file = opj(output_dir, f'{out}_stage-iter_constr-{constraints}_desc-prf_params.pkl')
    print(f'saving ...{iter_pkl_file}')
    # Dump everything!!! into pickle
    f = open(iter_pkl_file, "wb")
    pickle.dump(pkl_data, f)
    f.close()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
