#!/usr/bin/env python
'''Testing for track.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import numpy.testing as npt
import pdb
import unittest

from particle_tracking import track
from particle_tracking import readsnap

default_data_p = {
  'sdir' : './tests/test_data/test_data_with_new_id_scheme',
  'types' : [0,],
  'snap_ini' : 500,
  'snap_end' : 600,
  'snap_step' : 50,

  'outdir' : './tests/test_data/tracking_output',
  'tag' : 'test',
}

########################################################################

class TestConcatenateParticleData( unittest.TestCase ):

  def setUp( self ):

    self.id_finder = track.IDFinder()

    # The name of the function.
    self.fn = self.id_finder.concatenate_particle_data

  ########################################################################

  def test_basic( self ):
    '''Basically, does it work?'''

    # Input
    self.id_finder.sdir = './tests/test_data/test_data_with_new_id_scheme'
    self.id_finder.snum = 600
    self.id_finder.types = [0,]
    self.id_finder.target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
    self.id_finder.target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])

    self.id_finder.concatenate_particle_data()
    actual = self.id_finder.full_snap_data

    expected = {
      'id' : np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235]),
      'child_id' : np.array([1945060136, 0, 0, 938428052, 893109954, 0]),
      }

    for key in expected.keys():
      npt.assert_allclose( actual[key], expected[key] )

########################################################################

class TestSelectIDs( unittest.TestCase ):

  def setUp(self):

    self.id_finder = track.IDFinder()

    # The name of the function.
    self.fn = self.id_finder.select_ids

  ########################################################################

  def test_runs(self):
    '''Test that it even runs'''

    # Dummy data set
    self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
    self.id_finder.full_snap_data = {
      'id' : np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
      'rho' : np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
    }

    self.fn()

    assert True

  ########################################################################

  def test_works_simple( self ):
    '''Test that it runs in a simple case where there's no issues with duplicates.'''

    # Dummy data set
    self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
    self.id_finder.full_snap_data = {
      'id' : np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
      'rho' : np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
    }

    dfid = self.fn()

    expected = {
      'id' : np.array([ 38913508, 3211791, 10952235 ]),
      'rho' : np.array([ 3.45104532e-08,  6.80917722e-09, 1.54667816e-08 ]),
    }

    for key in dfid.keys():
      npt.assert_allclose( dfid[key], expected[key] )

  ########################################################################

  def test_works_duplicates( self ):

    # Dummy data set
    self.id_finder.target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
    self.id_finder.target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])
    self.id_finder.full_snap_data = {
      'id' : np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235]),
      'child_id' : np.array([1945060136, 0, 0, 938428052, 893109954, 0]),
      'rho' : np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
    }

    dfid = self.fn()

    expected = {
      'id' : self.id_finder.target_ids,
      'child_id' : self.id_finder.target_child_ids,
      'rho' : np.array([ 3.45104532e-08, 3.80374093e-10, 6.80917722e-09, 1.54667816e-08 ]),
    }

    for key in dfid.keys():
      npt.assert_allclose( dfid[key], expected[key] )

########################################################################

class TestFindIds( unittest.TestCase ):

  def setUp( self ):

    self.id_finder = track.IDFinder()

    # The name of the function.
    self.fn = self.id_finder.find_ids

  ########################################################################

  def test_basic( self ):
    '''Basically, does it work?.'''

    # Input
    sdir = './tests/test_data/test_data_with_new_id_scheme'
    snum = 600
    types = [0,]
    target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
    target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])

    # My knowledge, by hand
    target_inds = [4, 0, 1, 5]
    P = readsnap.readsnap( sdir, snum, 0, True, cosmological=True )

    expected = {
      'id' : target_ids,
      'child_id' : target_child_ids,
      'rho' : np.array([ P['rho'][ind] for ind in target_inds ]),
    }

    dfid, redshift, attrs = self.fn( sdir, snum, types, target_ids, \
                              target_child_ids=target_child_ids)

    for key in expected.keys():
      npt.assert_allclose( dfid[key], expected[key] )

    # Make sure the redshift's right too
    npt.assert_allclose( redshift, 0. )

########################################################################

class TestSaveTargetedParticles( unittest.TestCase ):

  def setUp( self ):

    self.particle_tracker = track.ParticleTracker( **default_data_p )

    # The name of the function.
    self.fn = self.particle_tracker.save_particle_tracks

  ########################################################################

  def test_runs( self ):

    self.fn()

  ########################################################################

  def test_basic( self ):

    self.fn()

    f = h5py.File( 'tests/test_data/tracking_output/ptrack_test.hdf5', 'r' )
    
    expected_snum = np.arange(600, 490, -50)
    actual_snum = f['snum'][...]
    npt.assert_allclose( expected_snum, actual_snum )

    #expected_rho_p0 =  np.array([  1.70068894e-08,   4.28708110e-09,   2.23610355e-09,
    #     5.92078259e-09,   6.38462647e-10,   6.44416458e-08,
    #     2.44035180e-06,   8.35424314e-09,   8.27433162e-10,
    #     2.15146115e-09,   1.94556549e-09])
    expected_rho_p0 =  np.array([  1.70068894e-08, 6.44416458e-08, 1.94556549e-09])
    actual_rho_p0 = f['rho'][...][0]
    npt.assert_allclose( expected_rho_p0, actual_rho_p0 )

    assert 'child_id' in f.keys()

  ########################################################################

  def test_has_attributes( self ):
    
    self.fn()

    f = h5py.File( 'tests/test_data/tracking_output/ptrack_test.hdf5', 'r' )

    # Load one of the original snapshots to compare
    P = readsnap.readsnap( 'tests/test_data/test_data_with_new_id_scheme', 600, 0, True, cosmological=True )
    
    compare_keys = [ 'omega_matter', 'omega_lambda', 'hubble' ]

    for key in compare_keys:
      npt.assert_allclose( P[key], f.attrs[key] )

    for key in default_data_p.keys():
      #npt.assert_allclose( default_data_p[key], f.attrs[key] )
      assert default_data_p[key] == f.attrs[key] 

  ########################################################################

  def test_get_target_ids( self ):

    # Remove any interfering attributes
    if hasattr( self.particle_tracker, 'target_ids'):
      del self.particle_tracker.target_ids

    self.particle_tracker.get_target_ids()

    expected = {
      'target_ids' : np.array([ 36091289, 36091289, 3211791, 10952235 ]),
      'target_child_ids' : np.array([ 893109954, 1945060136, 0, 0 ]),
    }

    for key in expected.keys():
      npt.assert_allclose( getattr( self.particle_tracker, key ), expected[key] )

