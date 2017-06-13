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
  'neg' : 1,
  }

default_ptrack = {
  'mt_gal_id' : np.array([
    [  0,  0,  2, -2, -2, ], # Merger, except in early snapshots
    [  0,  0,  0,  0, 0, ], # Always part of main galaxy
    [ -2,  0, -2, -2, -2, ], # CGM -> main galaxy -> CGM
    ]),
  'gal_id' : np.array([
    [  0,  2,  2, -2, -2, ], # Merger, except in early snapshots
    [  0,  0,  0,  0, 10, ], # Always part of main galaxy
    [ -2,  0, -2, -2, -2, ], # CGM -> main galaxy -> CGM
    ]),
  'snum' : np.array([ 600, 550, 500, 450, 10 ]),
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

    # Put in the number of snapshots and particles so that the function works correctly.
    self.classifier.n_snap = default_ptrack['gal_id'].shape[1]
    self.classifier.n_particles = default_ptrack['gal_id'].shape[0]

  ########################################################################

  def test_identify_if_in_galaxies( self ):

    expected_gal_event_id = np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ -1, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      ])

    # Run the function
    actual = self.classifier.identify_if_in_galaxies()

    npt.assert_allclose( expected_gal_event_id, actual )

  #########################################################################

  def test_identify_accretion( self ):

    expected = np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    # Get the prerequisites
    self.classifier.gal_event_id= np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ -1, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      ])

    # Run the function
    actual = self.classifier.identify_accretion()

    npt.assert_allclose( expected, actual )

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







