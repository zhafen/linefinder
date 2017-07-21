#!/usr/bin/env python

#SBATCH --job-name=worldline
#SBATCH --partition=normal
## Stampede node has 16 processors & 32 GB
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=0:05:00
#SBATCH --output=worldline_jobs/%j.out
#SBATCH --error=worldline_jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140023

'''For selecting IDs to track using worldline
This can be submitted as a batch job using: sbatch run_select_ids.py
Or it can simply be run in an interactive session with: ./run_select_ids.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import sys

import worldline.select_ids as select_ids

########################################################################
# Input Parameters
########################################################################

kwargs = {
  'snum_start' : 500,
  'snum_end' : 600,
  'snum_step' : 100,
  'ptypes' : [ 0, ],
  'out_dir' : './tests/data/tracking_output',
  'tag' : 'test',

  'snapshot_kwargs' : {
    'sdir' : './tests/data/stars_included_test_data',
    'load_additional_ids' : True,
    'ahf_index' : 600,
    'analysis_dir' : './tests/data/ahf_test_data',
    },
}

data_filters = [
  { 'data_key' : 'Den', 'data_min' : 1e-8, 'data_max' : 1e-7, },
]

########################################################################
# Run the ID Selecting
########################################################################

id_selector = select_ids.IDSelector( **kwargs )
id_selector.select_ids( data_filters )
