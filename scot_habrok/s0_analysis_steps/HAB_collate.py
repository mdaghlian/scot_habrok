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
    out = f"{sub}_{model}_{roi_fit}_{task}-fits"    

    for ib in np.arange(batch_num):
        ib_file = dag_find_file_in_folder(
            [sub, model, task, roi_fit, 'iter', constraints, '.pkl', 
            f'batch-{ib}-of-{batch_num}'],  # f'batch-{i:02}-of-{batch_num:03}'
            output_dir, 
            return_msg=None, 
            )  
        ib_new_file = ib_file.replace(
            f'batch-{ib}-of-{batch_num}',
            f'batch-{ib+1:03}-of-{batch_num:03}',
        )
        # os.system(f'mv {ib_file} {ib_new_file}')
        os.system(f'rm {ib_file}')
        # print(ib_new_file)        

    return


if __name__ == "__main__":
    main(sys.argv[1:])
