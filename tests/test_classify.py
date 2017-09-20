#!/usr/bin/env python
'''Testing for classify.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import unittest

import galaxy_diver.read_data.ahf as read_ahf
from pathfinder import classify

########################################################################
# Global Variables Useful for Testing
########################################################################

default_kwargs = {
  'ahf_data_dir' : './tests/data/ahf_test_data',
  'out_dir' : './tests/data/tracking_output',
  'tag' : 'test_classify',
  'neg' : 1,
  'wind_vel_min_vc' : 2.,
  'wind_vel_min' : 15.,
  'time_min' : 100., 
  'time_interval_fac' : 5.,
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
  'PType' : np.array([
    [  4,  4,  0,  0,  0, ], # Merger, except in early snapshots
    [  4,  0,  0,  0,  0, ], # Always part of main galaxy
    [  0,  0,  0,  0,  0, ], # CGM -> main galaxy -> CGM
    ]),
  'P' : np.array([
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
  'V' : np.array([
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
  'redshift' : np.array([ 0.        ,  0.06984665,  0.16946003, 0.28952773953090749, 12.310917860336163 ]),
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

    self.classifier = classify.Classifier( **default_kwargs )

  ########################################################################

  def test_basic( self ):

    self.classifier.read_data_files()

    expected = 1.700689e-08
    actual = self.classifier.ptrack['Den'][0,0]
    npt.assert_allclose( expected, actual )

  ########################################################################

class TestDerivedFunctions( unittest.TestCase ):

  def setUp( self ):

    self.classifier = classify.Classifier( **default_kwargs )

  ########################################################################

  def test_get_radial_velocity( self ):

    self.classifier.read_data_files()

    # Set the second particle at snapshot 550 to be at the center of the main halo at that redshift
    # Also set the velocity of the second particle at snapshot 550 to the velocity of the main halo at that redshift
    # This should result in an identically 0 radial velocity
    self.classifier.ptrack[ 'P' ][ 1, 1 ] = np.array([ 29372.26565053,  30929.16894187,  32415.81701217 ])
    self.classifier.ptrack[ 'P' ][ 1, 1 ] *= 1./(1. + self.classifier.ptrack['redshift'][ 1 ])/self.classifier.ptrack_attrs['hubble']
    self.classifier.ptrack[ 'V' ][ 1, 1 ] = np.array([ -49.05,  72.73,  96.86 ])

    # Get the result
    result = self.classifier.get_radial_velocity()

    # Make sure we have the right shape.
    assert result.shape == self.classifier.ptrack[ 'Den' ].shape

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

  def test_get_time_difference( self ):

    self.classifier.read_data_files()

    result = self.classifier.get_time_difference()

    # Expected difference in time, from NED's cosmology calculator.
    travel_time_at_snum_550 = 0.927*1e3 # In Myr
    travel_time_at_snum_600 = 2.104*1e3 # In Myr
    expected_0 = travel_time_at_snum_550
    expected_1 = travel_time_at_snum_600 - travel_time_at_snum_550

    npt.assert_allclose( expected_0, result[0][0], 1e-3)
    npt.assert_allclose( expected_1, result[1][1], 1e-3)

  ########################################################################


########################################################################

class TestIdentifyAccrectionEjectionAndMergers( unittest.TestCase ):

  def setUp( self ):

    self.classifier = classify.Classifier( **default_kwargs )

    # Emulate the loading data phase
    self.classifier.ptrack = default_ptrack
    self.classifier.ptrack_attrs = default_ptrack_attrs

    # Put in the number of snapshots and particles so that the function works correctly.
    self.classifier.n_snap = default_ptrack['gal_id'].shape[1]
    self.classifier.n_particle = default_ptrack['gal_id'].shape[0]

  ########################################################################

  def test_identify_is_in_other_gal( self ):

    self.classifier.ahf_reader = read_ahf.AHFReader( default_kwargs['ahf_data_dir'] )
    self.classifier.ahf_reader.get_mtree_halos( 'snum' )

    expected = np.array([
      [ 0, 1, 1, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    # Run the function
    actual = self.classifier.identify_is_in_other_gal()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_identify_is_in_main_gal( self ):

    # Prerequisites
    self.classifier.is_in_other_gal = np.array([
      [ 0, 1, 1, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    expected = np.array([
      [ 1, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 1, 1, 1, 1, 1, ], # Always part of main galaxy
      [ 0, 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    # Run the function
    actual = self.classifier.identify_is_in_main_gal()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_calc_gal_event_id( self ):

    # Prerequisite
    self.classifier.is_in_main_gal = np.array([
      [ 1, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 1, 1, 1, 1, 1, ], # Always part of main galaxy
      [ 0, 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    expected_gal_event_id = np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ -1, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      ])

    # Run the function
    actual = self.classifier.calc_gal_event_id()

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
    
  ########################################################################

  def test_identify_ejection( self ):

    # Prerequisites
    self.classifier.ahf_reader = read_ahf.AHFReader( default_kwargs['ahf_data_dir'] )
    self.classifier.ahf_reader.get_mtree_halos( 'snum' )

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

    # Run the function
    actual = self.classifier.identify_ejection()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_cum_num_acc( self ):

    self.classifier.is_accreted = np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      [ 1, 0, 1, 0, ], # Accreted twice
      ]).astype( bool )

    actual = self.classifier.get_cum_num_acc()
    expected = np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 1, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      [ 2, 1, 1, 0, ], # Accreted twice
      ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_redshift_first_acc( self ):

    self.classifier.is_before_first_acc = np.array([
      [ 0, 1, 1, 1, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 1, 1, ], # CGM -> main galaxy -> CGM
      [ 1, 1, 1, 1, ], # Never accreted
      ]).astype( bool )
    self.classifier.n_particle = 4

    expected = np.array([ 0., -1., 0.06984665, -1. ])
    actual = self.classifier.get_redshift_first_acc()
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_ind_first_acc( self ):

    self.classifier.is_before_first_acc = np.array([
      [ 0, 1, 1, 1, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 1, 1, ], # CGM -> main galaxy -> CGM
      [ 1, 1, 1, 1, ], # Never accreted
      ]).astype( bool )
    self.classifier.n_particle = 4

    expected = np.array([ 0, -99999, 1, -99999 ])
    actual = self.classifier.ind_first_acc
    npt.assert_allclose( expected, actual )

  #########################################################################


  ########################################################################

  def test_identify_is_before_first_acc( self ):

    # Prerequisites
    self.classifier.cum_num_acc = np.array([
      [ 1, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 1, 1, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_in_main_gal = np.array([
      [ 1, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 1, 1, 1, 1, 1, ], # Always part of main galaxy
      [ 0, 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    expected = np.array([
      [ 0, 1, 1, 1, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 1, 1, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    actual = self.classifier.identify_is_before_first_acc()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_time_in_other_gal_before_acc( self ):

    # Prerequisites
    self.classifier.dt = self.classifier.get_time_difference()
    self.classifier.is_before_first_acc = np.array([
      [ 0, 1, 1, 1, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 1, 1, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_in_other_gal = np.array([
      [ 0, 1, 1, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    # Calculated using NED cosmology calculator
    expected = np.array([
      2.404*1e3, # Merger, except in early snapshots
      0.,    # Always part of main galaxy
      0.,    # CGM -> main galaxy -> CGM
      ])

    actual = self.classifier.get_time_in_other_gal_before_acc()

    npt.assert_allclose( expected, actual, rtol=1e-3 )

  ########################################################################

  def test_get_time_in_other_gal_before_acc_during_interval( self ):
    '''Our interval is 200 Myr'''

    # Prerequisites
    # For this test we're not going to use the default data
    self.classifier.time_interval_fac = 1.5
    self.classifier.dt = np.array([
       [   50.,   50.,  50.,  50.,  50., ],
       [   50.,   50.,  50.,  50.,  50., ],
       [   50.,   50.,  50.,  50.,  50., ],
       [   50.,   50.,  50.,  50.,  50., ],
       [   50.,   50.,  50.,  50.,  50., ],
       [   50.,   50.,  50.,  50.,  50., ],
       ])
    self.classifier.is_before_first_acc = np.array([
      [ 0, 1, 1, 1, 1, ], # Merger, except in early snapshots
      [ 0, 1, 1, 1, 1, ], # Another merger
      [ 0, 1, 1, 1, 1, ], # Mass transfer
      [ 0, 0, 1, 1, 1, ], # Another test
      [ 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 0, 1, 1, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_in_other_gal = np.array([
      [ 0, 1, 1, 1, 1, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 1, 1, 1, ], # Another merger
      [ 0, 1, 0, 0, 1, 0, ], # Mass transfer
      [ 0, 0, 1, 0, 1, 1, ], # Another test
      [ 0, 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 0, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )
    # Correct the number of snapshots, accordingly
    self.classifier.n_snap = self.classifier.is_in_other_gal.shape[1]

    expected = np.array([
      150., # Merger, except in early snapshots
      50., # Another merger
      50.,    # Mass transfer
      100.,  # Another test
      0.,    # Always part of main galaxy
      0.,    # CGM -> main galaxy -> CGM
      ])

    actual = self.classifier.get_time_in_other_gal_before_acc_during_interval()

    npt.assert_allclose( expected, actual, rtol=1e-3 )

  #########################################################################

  def test_identify_pristine( self ):

    # Prerequisites
    self.classifier.time_in_other_gal_before_acc = np.array([
      2.404*1e3, # Merger, except in early snapshots
      0.,    # Always part of main galaxy
      0.,    # CGM -> main galaxy -> CGM
      ])
    self.classifier.is_in_main_gal = np.array([
      [ 1, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 1, 1, 1, 1, 1, ], # Always part of main galaxy
      [ 0, 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    expected = np.array([
      0,    # Merger, except in early snapshots
      1,    # Always part of main galaxy
      1,    # CGM -> main galaxy -> CGM
      ]).astype( bool )

    actual = self.classifier.identify_pristine()

    npt.assert_allclose( expected, actual, )

  #########################################################################

  def test_identify_preprocessed( self ):

    # Prerequisites
    self.classifier.time_in_other_gal_before_acc = np.array([
      2.404*1e3, # Merger, except in early snapshots
      0.,    # Always part of main galaxy
      0.,    # CGM -> main galaxy -> CGM
      ])
    self.classifier.is_in_main_gal = np.array([
      [ 1, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 1, 1, 1, 1, 1, ], # Always part of main galaxy
      [ 0, 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )

    expected = np.array([
      1,    # Merger, except in early snapshots
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )

    actual = self.classifier.identify_preprocessed()

    npt.assert_allclose( expected, actual, )

  #########################################################################

  def test_identify_mass_transfer( self ):

    # Prerequisites
    self.classifier.is_preprocessed = np.array([
      1,    # Merger, except in early snapshots
      1,    # Mass transfer
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.time_in_other_gal_before_acc_during_interval = np.array([
      300.,    # Merger, except in early snapshots
      50.,    # Mass transfer
      0.,    # Always part of main galaxy
      0.,    # CGM -> main galaxy -> CGM
      ])

    expected = np.array([
      0,    # Merger, except in early snapshots
      1,    # Mass Transfer
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )

    actual = self.classifier.identify_mass_transfer()

    npt.assert_allclose( expected, actual, )

  #########################################################################

  def test_identify_merger( self ):

    # Prerequisites
    self.classifier.is_preprocessed = np.array([
      1,    # Merger, except in early snapshots
      1,    # Mass transfer
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.time_in_other_gal_before_acc_during_interval = np.array([
      300.,    # Merger, except in early snapshots
      50.,    # Mass transfer
      0.,    # Always part of main galaxy
      0.,    # CGM -> main galaxy -> CGM
      ])

    expected = np.array([
      1,    # Merger, except in early snapshots
      0,    # Mass Transfer
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )

    actual = self.classifier.identify_merger()

    npt.assert_allclose( expected, actual, )

  #########################################################################

  def test_identify_wind( self ):

    # Prerequisites
    self.classifier.is_ejected = np.array([
      [ 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      [ 1, 0, 1, 0, ], # Another test case
      ]).astype( bool )
    self.classifier.n_particle = self.classifier.is_ejected.shape[0]

    expected = np.array([
      [ 0, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 1, 0, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      [ 1, 1, 1, 0, 0, ], # Another test case
      ]).astype( bool )

    actual = self.classifier.identify_wind()

    npt.assert_allclose( expected, actual, )

########################################################################
########################################################################

class TestFullClassifierPipeline( unittest.TestCase ):

  def setUp( self ):
    # Mock the code version so we don't repeatedly change test data
    patcher = patch( 'galaxy_diver.utils.utilities.get_code_version' )
    self.addCleanup( patcher.stop )
    self.mock_code_version = patcher.start()

    self.classifier = classify.Classifier( **default_kwargs )

    self.savefile = './tests/data/tracking_output/classifications_test_classify.hdf5'
    self.events_savefile = './tests/data/tracking_output/events_test_classify.hdf5'

    # Because we're skipping this step, we need to make sure we're not tossing objects around
    self.classifier.ptracks_tag = self.classifier.tag
    self.classifier.galids_tag = self.classifier.tag

    if os.path.isfile( self.savefile ):
      os.system( 'rm {}'.format( self.savefile ) )
    if os.path.isfile( self.events_savefile ):
      os.system( 'rm {}'.format( self.events_savefile ) )

  ########################################################################

  def tearDown( self ):

    if os.path.isfile( self.savefile ):
      os.system( 'rm {}'.format( self.savefile ) )
    if os.path.isfile( self.events_savefile ):
      os.system( 'rm {}'.format( self.events_savefile ) )

  ########################################################################

  def test_save_classifications( self ):

    # Give it filenames to save.
    self.classifier.ptrack_filename = 'test_ptrack_filename'
    self.classifier.galfind_filename = 'test_galfind_filename'

    self.classifier.ahf_reader = read_ahf.AHFReader( default_kwargs['ahf_data_dir'] )

    # Prerequisites
    self.classifier.is_pristine = np.array([
      0,    # Merger, except in early snapshots
      0,    # Mass Transfer
      1,    # Always part of main galaxy
      1,    # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_preprocessed = np.array([
      1,    # Merger, except in early snapshots
      1,    # Mass Transfer
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_mass_transfer = np.array([
      0,    # Merger, except in early snapshots
      1,    # Mass Transfer
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_merger = np.array([
      1,    # Merger, except in early snapshots
      0,    # Mass Transfer
      0,    # Always part of main galaxy
      0,    # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_wind = np.array([
      [ 0, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 1, 0, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      [ 1, 1, 1, 0, 0, ], # Another test case
      ]).astype( bool )
    self.classifier.redshift_first_acc = np.array([ 0., -1., 0.06984665, -1. ])

    # Change values from defaults so that we save without issue
    self.classifier.halo_file_tag = 'smooth'
    self.classifier.mtree_halos_index = None

    # The function itself.
    self.classifier.save_classifications( self.classifier.classifications_to_save )

    # Try to load the data
    f = h5py.File( self.savefile, 'r')
    
    npt.assert_allclose( self.classifier.is_pristine, f['is_pristine'][...] )

  ########################################################################

  def test_save_events( self ):

    self.classifier.is_ejected = np.array([
      [ 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, ], # Always part of main galaxy
      [ 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      [ 1, 0, 1, 0, ], # Another test case
      ]).astype( bool )
    self.classifier.is_in_other_gal = np.array([
      [ 0, 1, 1, 0, 0, ], # Merger, except in early snapshots
      [ 0, 0, 0, 0, 0, ], # Always part of main galaxy
      [ 0, 0, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.is_in_main_gal = np.array([
      [ 1, 0, 0, 0, 0, ], # Merger, except in early snapshots
      [ 1, 1, 1, 1, 1, ], # Always part of main galaxy
      [ 0, 1, 0, 0, 0, ], # CGM -> main galaxy -> CGM
      ]).astype( bool )
    self.classifier.redshift_first_acc = np.array([ 0., np.nan, 0.06984665, 0.16946003 ])
    self.classifier._ind_first_acc = np.array([ 0, -99999, 1, -99999 ])

    # Change values so that we save without issue
    self.classifier.halo_file_tag = 'smooth'
    self.classifier.mtree_halos_index = None

    self.classifier.save_events( self.classifier.events_to_save )

    f = h5py.File( self.events_savefile, 'r' )

    for event_type in [ 'is_ejected', 'is_in_other_gal', 'is_in_main_gal', 'redshift_first_acc', ]:
      assert event_type in f.keys()

  ########################################################################

  def test_full_pipeline( self ):
    '''Test that we can run the full pipeline from just the files.'''

    self.classifier.classify_particles()

    expected = np.array([
      1,
      1,
      1,
      1,
      ]).astype( bool )

    f = h5py.File( self.savefile, 'r')
    actual =  f['is_pristine'][...]

    # Make sure that we've saved our input arguments
    for key in default_kwargs.keys():
      assert default_kwargs[key] == f['parameters'].attrs[key]
