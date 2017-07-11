#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import galaxy_diver.read_data.ahf as read_ahf
from worldline import galaxy_find

########################################################################
# Useful global test variables
########################################################################

gal_finder_kwargs = {
  'redshift' : 0.16946003,
  'snum' : 500,
  'hubble' : 0.70199999999999996,
  'sdir' : './tests/test_data/ahf_test_data',
  'mtree_halos_index' : 600,
}

ptrack_gal_finder_kwargs = {
  'sdir' : './tests/test_data/ahf_test_data',
  'tracking_dir' : './tests/test_data/tracking_output',
  'tag' : 'test',
  'mtree_halos_index' : 600,
}

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################

class TestGalaxyFinder( unittest.TestCase ):

  def setUp( self ):

    # Get input data
    comoving_halo_coords = np.array([ [ 29414.96458784,  30856.75007114,  32325.90901812],
                                      [ 31926.42103071,  51444.46756529,   1970.1967437 ] ])

    self.redshift = gal_finder_kwargs['redshift']
    self.hubble = gal_finder_kwargs['hubble']
    halo_coords = comoving_halo_coords/(1. + self.redshift)/self.hubble

    # Make the necessary kwargs
    self.kwargs = gal_finder_kwargs

    self.galaxy_finder = galaxy_find.GalaxyFinder( halo_coords, **self.kwargs ) 

    # Get the necessary reader.
    self.galaxy_finder.ahf_reader = read_ahf.AHFReader( self.kwargs['sdir'] )
    
    # Get the full needed ahf info.
    self.galaxy_finder.ahf_reader.get_halos( 500 )

  ########################################################################

  def test_find_containing_halos( self ):

    result = self.galaxy_finder.find_containing_halos()

    # If none of the distances are within any of the halos, we have a problem.
    assert result.sum() > 0

  ########################################################################

  def test_find_containing_halos_strict( self ):
    '''Here I'll restrict the fraction to a very small fraction of the virial radius, such that the sum of the results should be two.
    '''

    result = self.galaxy_finder.find_containing_halos( 0.0001 )

    # If none of the distances are within any of the halos, we have a problem.
    npt.assert_allclose( 2, result.sum() )

  ########################################################################

  def test_find_containing_halos_r_scale( self ):
    '''Test that this works for using r_scale.'''

    # Set the length scale
    self.galaxy_finder.length_scale = 'r_scale'

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29414.96458784 + ,  30856.75007114,  32325.90901812], # Just outside the scale radius of mt halo 0 at snap 500.
      ])

    result = self.galaxy_finder.find_containing_halos( 1. )

  ########################################################################

  def test_find_mt_containing_halos( self ):
    
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                           # mt halo 9 is 0.5 R_vir_mt_0 (2 R_vir_mt_9) away from the center of mt halo 0
      [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble

    actual = self.galaxy_finder.find_mt_containing_halos( 2.5 )

    # Build the expected output
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], 6) ).astype( bool )
    expected[ 0, 0 ] = True
    expected[ 0, -2 ] = True
    expected[ 1, 0 ] = True
    expected[ 1, -2 ] = True
    expected[ 2, -1 ] = True

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_mt_containing_halos_r_scale( self ):

    assert False, "Need to do."

  ########################################################################

  def test_find_smallest_host_halo( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812],
      [ 31926.42103071,  51444.46756529,   1970.1967437 ],
      [ 29467.07226789,  30788.6179313 ,  32371.38749237],
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble

    self.galaxy_finder.n_particles = 4

    expected = np.array( [0, 6962, 7, 3783] )
    actual = self.galaxy_finder.find_halo_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_smallest_host_halo_none( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      ])

    expected = np.array( [-2, -2] )
    actual = self.galaxy_finder.find_halo_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_host_id( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble

    self.galaxy_finder.n_particles = 3

    expected = np.array( [-1, 1, 3610] )
    actual = self.galaxy_finder.find_host_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_host_id_none( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      ])

    expected = np.array( [-2, -2] )
    actual = self.galaxy_finder.find_host_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_mt_halo_id( self ):
    
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                           # mt halo 9 is 0.5 R_vir_mt_0 (2 R_vir_mt_9) away from the center of mt halo 0
      [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
      [             0.,              0.,              0.], # The middle of nowhere.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble
    self.galaxy_finder.n_particles = 4

    actual = self.galaxy_finder.find_halo_id( 2.5, 'mt_halo_id' )

    # Build the expected output
    expected = np.array([ 0, 0, 19, -2 ])

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_mt_halo_id_early_universe( self ):
    '''Test that, when there are no galaxies formed, we return an mt halo value of -2'''
    
    # Set it to early redshifts
    self.galaxy_finder.kwargs['snum'] = 0

    # It doesn't really matter where the particles are, because there shouldn't be any galaxies anyways....
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                           # mt halo 9 is 0.5 R_vir_mt_0 (2 R_vir_mt_9) away from the center of mt halo 0
      [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
      [             0.,              0.,              0.], # The middle of nowhere.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + 30.)/self.hubble
    self.galaxy_finder.n_particles = 4

    actual = self.galaxy_finder.find_halo_id( 2.5, 'mt_halo_id' )

    # Build the expected output
    expected = np.array([ -2, -2, -2, -2 ])

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_ids( self ):

    particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    particle_positions *= 1./(1. + self.redshift)/self.hubble

    expected = {
      'host_halo_id' : np.array( [-1, 1, 3610] ),
      'halo_id' : np.array( [0, 10, 3783] ),
      'host_gal_id' : np.array( [-1, 1, 3610] ),
      'gal_id' : np.array( [0, 10, 3783] ),
      'mt_gal_id' : np.array( [0, -2, -2] ),
      'mt_halo_id' : np.array( [0, 1, 0] ),
    }

    # Do the actual calculation
    galaxy_finder = galaxy_find.GalaxyFinder( particle_positions, **self.kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
      npt.assert_allclose( expected[key], actual[key] )

  ########################################################################

  def test_find_ids_snap0( self ):

    particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    particle_positions *= 1./(1. + 30.)/self.hubble

    expected = {
      'host_halo_id' : np.array( [-2, -2, -2] ),
      'halo_id' : np.array( [-2, -2, -2] ),
      'host_gal_id' : np.array( [-2, -2, -2] ),
      'gal_id' : np.array( [-2, -2, -2] ),
      'mt_gal_id' : np.array( [-2, -2, -2] ),
      'mt_halo_id' : np.array( [-2, -2, -2] ),
    }

    # Setup the input parameters
    snap0_kwargs = copy.deepcopy( self.kwargs )
    snap0_kwargs['snum'] = 0

    # Do the actual calculation
    galaxy_finder = galaxy_find.GalaxyFinder( particle_positions, **snap0_kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
      npt.assert_allclose( expected[key], actual[key] )

  ########################################################################

  def test_find_ids_early_universe( self ):

    particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    particle_positions *= 1./(1. + 28.)/self.hubble

    expected = {
      'host_halo_id' : np.array( [-2, -2, -2] ),
      'halo_id' : np.array( [-2, -2, -2] ),
      'host_gal_id' : np.array( [-2, -2, -2] ),
      'gal_id' : np.array( [-2, -2, -2] ),
      'mt_gal_id' : np.array( [-2, -2, -2] ),
      'mt_halo_id' : np.array( [-2, -2, -2] ),
    }

    # Setup the input parameters
    snap0_kwargs = copy.deepcopy( self.kwargs )
    snap0_kwargs['snum'] = 1

    # Do the actual calculation
    galaxy_finder = galaxy_find.GalaxyFinder( particle_positions, **snap0_kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
      npt.assert_allclose( expected[key], actual[key] )

  ########################################################################

  def test_pass_ahf_reader( self ):
    '''Test that it still works when we pass in an ahf_reader. '''

    particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    particle_positions *= 1./(1. + self.redshift)/self.hubble

    expected = {
      'host_halo_id' : np.array( [-1, 1, 3610] ),
      'halo_id' : np.array( [0, 10, 3783] ),
      'host_gal_id' : np.array( [-1, 1, 3610] ),
      'gal_id' : np.array( [0, 10, 3783] ),
      'mt_gal_id' : np.array( [0, -2, -2] ),
      'mt_halo_id' : np.array( [0, 1, 0] ),
    }

    # Prepare an ahf_reader to pass along.
    ahf_reader = read_ahf.AHFReader( self.kwargs['sdir'] )

    # Muck it up by making it try to retrieve data
    ahf_reader.get_halos( 600 )
    ahf_reader.get_mtree_halos( 600, tag='smooth' )

    # Do the actual calculation
    galaxy_finder = galaxy_find.GalaxyFinder( particle_positions, ahf_reader=ahf_reader, **self.kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
      npt.assert_allclose( expected[key], actual[key] )

########################################################################

class TestParticleTrackGalaxyFinder( unittest.TestCase ):

  def setUp( self ):

    self.originalfile = './tests/test_data/tracking_output/ptrack_test.hdf5'
    self.savefile = './tests/test_data/tracking_output/galfind_test.hdf5'

  ########################################################################

  def tearDown( self ):

    os.system( 'rm {}'.format( self.savefile ) )

  ########################################################################

  @slow
  def test_find_galaxies_for_particle_tracks( self ):

    particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder( **ptrack_gal_finder_kwargs )
    particle_track_gal_finder.find_galaxies_for_particle_tracks()

    f = h5py.File( self.savefile, 'r' )

    # What we expect (calculated using the positions of the particles in snum 500 )
    particle_positions = np.array([
       [ 34463.765625  ,  37409.45703125,  38694.92578125],
       [ 35956.61328125,  38051.79296875,  39601.91796875],
       [ 34972.05078125,  39093.890625  ,  39445.94921875],
       [ 35722.984375  ,  38222.14453125,  39245.52734375],
    ])
    galaxy_finder = galaxy_find.GalaxyFinder( particle_positions, **gal_finder_kwargs )
    expected = galaxy_finder.find_ids()

    for key in expected.keys():
      npt.assert_allclose( expected[key], f[key][...][:,-1] )

    # Make sure the main MT halo ID is the one we expect.
    assert f.attrs['main_mt_halo_id'] == 0

    # Make sure we have stored the data parameters too.
    for key in ptrack_gal_finder_kwargs.keys():
      assert ptrack_gal_finder_kwargs[key] == f.attrs[key]

    f.close()
