#!/usr/bin/env python

#SBATCH --job-name=worldline
#SBATCH --partition=largemem
## Stampede node has 16 processors & 32 GB
## Except largemem nodes, which have 32 processors & 1 TB
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --output=worldline_jobs/%j.out
#SBATCH --error=worldline_jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140023

'''For selecting IDs to track using worldline
This can be submitted as a batch job using: sbatch run_select.py
Or it can simply be run in an interactive session with: ./run_select.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import sys

import worldline.select as select

########################################################################
# Put Necessary Global Variables Here
########################################################################

run_select = True
run_sample_ids = True

sdir = '/scratch/03057/zhafen/m12v_mr_Dec5_2013_3'
out_dir = '/work/03057/zhafen/worldline_data/m12v_mr_Dec5_2013_3'
tag = 'm12iF1_newids'

########################################################################
# Configuration Parameters
########################################################################

### Parameters for ID Selection ###
kwargs = {
  'snum_start' : 30,
  'snum_end' : 440,
  'snum_step' : 1,
  'ptypes' : [ 0, 4, ],
  'out_dir' : out_dir,
  'tag' : tag,
  'n_processors' : 32,

  'snapshot_kwargs' : {
    'sdir' : sdir,
    'load_additional_ids' : False,
    'ahf_index' : 440,
    'analysis_dir' : sdir,
    },
}

data_filters = [
  { 'data_key' : 'Rf', 'data_min' : 0., 'data_max' : 1., },
]

### Parameters for ID Sampling ###

sampler_kwargs = {
  'sdir' : out_dir,
  'tag' : tag,
}

########################################################################
# Run the ID Selecting
########################################################################

if run_select:
  id_selector = select.IDSelector( **kwargs )
  id_selector.select_ids( data_filters )

########################################################################
# Run the ID Sampling
########################################################################

if run_sample_ids:
  id_sampler = select.IDSampler( **sampler_kwargs )
  id_sampler.sample_ids()
