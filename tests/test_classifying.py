#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import numpy.testing as npt
import pdb
import unittest

from particle_tracking import classifying

########################################################################
# Global Variables Useful for Testing
########################################################################

default_data_p = {
  'sdir' : './tests/test_data/ahf_test_data',
  'tracking_dir' : './tests/test_data/tracking_output',
  'tag' : 'test',
  }

default_ptrack = {
  'mt_gal_id' : np.array([
    [ -2, 2, 2, 0, ], # Merger, except in early snapshots
    [ 0, 0, 0, 0, ], # Always part of main galaxy
    [ -2, -2, 0, -2, ], # CGM -> Halo -> CGM
    ]),
  }

default_ptrack_attrs = {
  'main_mt_halo_id' : 0,
  }

########################################################################
# Test Cases
########################################################################

class TestReadPTrack( unittest.TestCase ):

  def setUp( self ):

    self.classifier = classifying.Classifier( default_data_p )

  ########################################################################

  def test_basic( self ):

    self.classifier.read_data_files()

    expected = 1.700689e-08
    actual = self.classifier.ptrack['rho'][0,0]
    npt.assert_allclose( expected, actual )

########################################################################

class TestIdentifyAccrectionEjectionAndMergers( unittest.TestCase ):

  def setUp( self ):

    self.classifier = classifying.Classifier( default_data_p )

    self.classifier.ptrack = default_ptrack

    self.classifier.ptrack_attrs = default_ptrack_attrs

    # Put in the number of snapshots so that the function works correctly.
    self.classifier.n_snap = 4

  ########################################################################

  def test_identify_if_in_galaxies( self ):

    #DEBUG
    import pdb; pdb.set_trace()

    # Run the function
    self.classifier.identify_if_in_galaxies()

    expected_gal_event_id = np.array([
        [ 0, 0, 1, ],
        [ 0, 0, 0, ],
        [ 0, 1, -1, ],
        ])

    npt.assert_allclose( expectget_timeed_gal_event_id, self.classifiers.gal_event_id )

  #########################################################################

  #def test_identify_accretion( self ):

  #  assert False, "Need to do this test."

  #########################################################################

  #def test_identify_ejection( self ):

  #  assert False, "Need to do this test."

  #########################################################################

  #def test_get_time_in_galaxies( self ):

  #  assert False, "Need to do this test."

  #########################################################################
  #def test_identify_pristine( self ):

  #  assert False, "Need to do this test."

  #########################################################################

  #def test_identify_preprocessed( self ):

  #  assert False, "Need to do this test."

  #########################################################################

  #def test_identify_mass_transfer( self ):

  #  assert False, "Need to do this test."

  #########################################################################

  #def test_identify_merger( self ):

  #  assert False, "Need to do this test."

  #########################################################################

  #def test_identify_wind( self ):

  #  assert False, "Need to do this test."

  #########################################################################







