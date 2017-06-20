#!/usr/bin/env python
'''Script for tracking particles.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import galaxy_finding
import tracking

########################################################################
# Input Parameters
########################################################################

kwargs = {
  'sdir' : '../tests/test_data/test_data_with_new_id_scheme',
  'types' : [0,],
  'snap_ini' : 500,
  'snap_end' : 600,
  'snap_step' : 50,

  'target_ids' : np.array([ 36091289, 36091289, 3211791, 10952235 ]),
  'target_child_ids' : np.array([ 893109954, 1945060136, 0, 0 ]),
  'outdir' : '../tests/test_data/tracking_output',
  'tag' : 'test_classify',
}

########################################################################
# Input Parameters for the Galaxy Finder
########################################################################

# Most (if not all) of the input parameters should be taken directly from the original ptrack kwargs
gal_finder_kwargs = {
  'sdir' : '../tests/test_data/ahf_test_data',
  'tracking_dir' : kwargs['outdir'],
  'tag' : kwargs['tag'],
}

########################################################################
# Run the Tracking
########################################################################

particle_tracker = tracking.ParticleTracker( **kwargs )
particle_tracker.save_particle_tracks()

########################################################################
# Run the Galaxy Finding
########################################################################

particle_track_gal_finder = galaxy_finding.ParticleTrackGalaxyFinder( **gal_finder_kwargs )
particle_track_gal_finder.find_galaxies_for_particle_tracks()
