#!/usr/bin/env python
'''Script for tracking particles.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import tracking

########################################################################
# Input Parameterss
########################################################################

data_p = {
  'sdir' : '/home1/03057/zhafen/repos/worldline/tests/test_data/test_data_with_new_id_scheme',
  'types' : [0,],
  'snap_ini' : 500,
  'snap_end' : 600,
  'snap_step' : 10,

  'target_ids' : np.array([ 36091289, 36091289, 3211791, 10952235 ]),
  'target_child_ids' : np.array([ 893109954, 1945060136, 0, 0 ]),
  'outdir' : '/home1/03057/zhafen/repos/worldline/tests/test_data/tracking_output',
  'tag' : 'test',
}

########################################################################
# Run the Tracking
########################################################################

id_finder_full = tracking.IDFinderFull( data_p )
id_finder_full.save_target_particles()
