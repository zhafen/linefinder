#!/usr/bin/env python

#SBATCH --job-name=tracking
#SBATCH --partition=development
## Stampede node has 16 processors & 32 GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=4
#SBATCH --time=0:05:00
#SBATCH --output=tracking_jobs/%j.out
#SBATCH --error=tracking_jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140023
##SBATCH --begin=now+48hour

'''Script for tracking particles.
This can be submitted as a batch job using: sbatch run_tracking.py
Or it can simply be run in an interactive session with: ./run_tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

from particle_tracking import galaxy_find
from particle_tracking import track

########################################################################
# Input Parameters
########################################################################

kwargs = {
  'sdir' : '../tests/test_data/test_data_with_new_id_scheme',
  'types' : [0,],
  'snap_ini' : 500,
  'snap_end' : 600,
  'snap_step' : 50,

  'outdir' : '../tests/test_data/tracking_output',
  'tag' : 'test',
}

########################################################################
# Input Parameters for the Galaxy Finder
########################################################################

# Most (if not all) of the input parameters should be taken directly from the original ptrack kwargs
gal_finder_kwargs = {
  'sdir' : '../tests/test_data/ahf_test_data',
  'tracking_dir' : kwargs['outdir'],
  'tag' : kwargs['tag'],
  'mtree_halos_index' : 'snum',
}

########################################################################
# Run the Tracking
########################################################################

particle_tracker = track.ParticleTracker( **kwargs )
particle_tracker.save_particle_tracks()

########################################################################
# Run the Galaxy Finding
########################################################################

particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder( **gal_finder_kwargs )
particle_track_gal_finder.find_galaxies_for_particle_tracks()
