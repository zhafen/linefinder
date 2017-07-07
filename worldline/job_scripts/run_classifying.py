#!/usr/bin/env python
'''Script for classifying particles.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import classifying

########################################################################
# Input Parameters
########################################################################

kwargs = {
  'sdir' : '../tests/test_data/ahf_test_data',
  'tracking_dir' : '../tests/test_data/tracking_output',
  'tag' : 'test_classify',
  'neg' : 1,
  'wind_vel_min_vc' : 2.,
  'wind_vel_min' : 15.,
  'time_min' : 100., 
  'time_interval_fac' : 5.,
  }

########################################################################
# Run the Classifying
########################################################################

classifier = classifying.Classifier( **kwargs )
classifier.classify_particles()
