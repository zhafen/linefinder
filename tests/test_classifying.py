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

from particle_tracking import ahf_reading
from particle_tracking import classifying

########################################################################
# Global Variables Useful for Testing
########################################################################

default_data_p = {
  'sdir' : './tests/test_data/ahf_test_data',
  'tracking_dir' : './tests/test_data/tracking_output',
  'tag' : 'test_classify',
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

class TestDerivedFunctions( unittest.TestCase ):

  def setUp( self ):

    self.classifier = classifying.Classifier( default_data_p )

  ########################################################################

  def test_get_radial_velocity( self ):

    self.classifier.read_data_files()

    # Set the second particle at snapshot 550 to be at the center of the main halo at that redshift
    # Also set the velocity of the second particle at snapshot 550 to the velocity of the main halo at that redshift
    # This should result in an identically 0 radial velocity
    self.classifier.ptrack[ 'p' ][ 1, 1 ] = np.array([ 29372.26565053,  30929.16894187,  32415.81701217 ])
    self.classifier.ptrack[ 'p' ][ 1, 1 ] *= 1./(1. + self.classifier.ptrack['redshift'][ 1 ])/self.classifier.ptrack_attrs['hubble']
    #self.classifier.ptrack[ 'p' ][ 1, 1 ] = np.array([ 39109.18838863,  41182.20600492,  43161.67875451]) #This has already been converted to physical
    self.classifier.ptrack[ 'v' ][ 1, 1 ] = np.array([ -49.05,  72.73,  96.86 ])

    # Get the result
    result = self.classifier.get_radial_velocity()

    # Make sure we have the right shape.
    assert result.shape == self.classifier.ptrack[ 'rho' ].shape

    # Make sure that we have 0 radial velocity when we should
    npt.assert_allclose( result[ 1, 1 ], 0., atol=1e-3 )


########################################################################

class TestIdentifyAccrectionEjectionAndMergers( unittest.TestCase ):

  def setUp( self ):

    self.classifier = classifying.Classifier( default_data_p )

    # Emulate the loading data phase
    self.classifier.ptrack = default_ptrack
    self.classifier.ptrack_attrs = default_ptrack_attrs
    self.classifier.ahf_reader = ahf_reading.AHFReader( default_data_p['sdir'] )
    self.classifier.ahf_reader.get_mtree_halos( 'snum' )

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

  # TODO
  #def test_identify_ejection( self ):

  #  expected = np.array([
  #    [ 0, 0, 0, 0, ], # Merger, except in early snapshots
  #    [ 0, 0, 0, 0, ], # Always part of main galaxy
  #    [ 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
  #    ]).astype( bool )

  #  # Get the prerequisites
  #  self.classifier.gal_event_id= np.array([
  #    [ 1, 0, 0, 0, ], # Merger, except in early snapshots
  #    [ 0, 0, 0, 0, ], # Always part of main galaxy
  #    [ -1, 1, 0, 0, ], # CGM -> main galaxy -> CGM
  #    ])

  #  # Run the function
  #  actual = self.classifier.identify_ejection()

  #  npt.assert_allclose( expected, actual )

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







