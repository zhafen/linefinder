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
  'wind_vel_min_vc' : 2.,
  'wind_vel_min' : 15.,
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
  'Ptype' : np.array([
    [  4,  4,  0,  0,  0, ], # Merger, except in early snapshots
    [  4,  0,  0,  0,  0, ], # Always part of main galaxy
    [  0,  0,  0,  0,  0, ], # CGM -> main galaxy -> CGM
    ]),
  'p' : np.array([
    [ [ 41792.1633    ,  44131.2309735 ,  46267.67030708 ], # Merger, except in early snapshots
      [ 38198.04856455,  42852.63974461,  43220.86278364 ],
      [ 34972.28497249,  39095.17772698,  39446.83170768 ],
      [             0.,              0.,              0. ],
      [             0.,              0.,              0. ], ],
    [ [ 41792.1633    ,  44131.2309735 ,  46267.67030708 ], # Always part of main galaxy
      [ 39109.18846174,  41182.20608191,  43161.6788352  ],
      [ 35829.91969126,  37586.13659658,  39375.69670048 ],
      [ 32543.5697382 ,  33981.19081307,  35583.36876478 ],
      [  3245.25202392,   3136.94192456,   3317.2277023  ], ],
    [ [             0.,              0.,              0. ], # CGM -> main galaxy -> CGM
      [ 39109.18846174,  41182.20608191,  43161.6788352  ],
      [             0.,              0.,              0. ],
      [             0.,              0.,              0. ],
      [             0.,              0.,              0. ], ],
    ]),
  'v' : np.array([
    [ [-48.53,  72.1 ,  96.12], # Merger, except in early snapshots
      [-23.75,  91.13,  80.57],
      [-20.92,  92.55,  75.86],
      [-17.9 ,  92.69,  70.54],
      [ 0.,   0.,   0., ], ],
    [ [-48.53,  72.1 ,  96.12], # Always part of main galaxy
      [-49.05,  72.73,  96.86],
      [-48.89,  73.77,  97.25],
      [-49.75,  75.68,  96.52],
      [-12.43,  39.47,  13.  ], ],
    [ [-48.53 + 100.,  72.1 ,  96.12], # CGM -> main galaxy -> CGM
      [-49.05,  72.73,  96.86],
      [-48.89,  73.77,  97.25],
      [-49.75,  75.68,  96.52],
      [-12.43,  39.47,  13.  ], ],
    ]),
  'snum' : np.array([ 600, 550, 500, 450, 10 ]),
  'redshift' : np.array([ 0.        ,  0.06984665,  0.16946003, 0.290, 12.311 ]),
  }

default_ptrack_attrs = {
  'main_mt_halo_id' : 0,
  'hubble' : 0.70199999999999996,
  'omega_lambda' : 0.72799999999999998,
  'omega_matter' : 0.27200000000000002,
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
    self.classifier.ptrack[ 'v' ][ 1, 1 ] = np.array([ -49.05,  72.73,  96.86 ])

    # Get the result
    result = self.classifier.get_radial_velocity()

    # Make sure we have the right shape.
    assert result.shape == self.classifier.ptrack[ 'rho' ].shape

    # Make sure that we have 0 radial velocity when we should
    npt.assert_allclose( result[ 1, 1 ], 0., atol=1e-3 )

  ########################################################################

  def test_get_circular_velocity( self ):

    self.classifier.read_data_files()

    # What our actual circular velocity is
    result = self.classifier.get_circular_velocity()

    # Make sure we have the right dimensions
    assert result.shape == ( 3, )

    # We expect the circular velocity of a 1e12 Msun galaxy to be roughly ~100 km/s
    expected = 100.
    actual = result[600]
    npt.assert_allclose( expected, actual, rtol=0.5 )

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

  def test_identify_ejection( self ):

    expected = np.array([
      [ 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    # Get the prerequisites
    self.classifier.gal_event_id= np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ -1, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      ])

    #DEBUG
    import pdb; pdb.set_trace()

    # Run the function
    actual = self.classifier.identify_ejection()

    npt.assert_allclose( expected, actual )

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







