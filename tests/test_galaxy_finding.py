#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import numpy.testing as npt
import os
import pdb
import unittest

from particle_tracking import ahf_reading
from particle_tracking import galaxy_finding

########################################################################
# Useful global test variables
########################################################################

gal_finder_data_p = {
  'redshift' : 0.16946003,
  'snum' : 500,
  'hubble' : 0.70199999999999996,
  'sdir' : './tests/test_data/ahf_test_data',
}

ptrack_gal_finder_data_p = {
  'sdir' : './tests/test_data/ahf_test_data',
  'tracking_dir' : './tests/test_data/tracking_output',
  'tag' : 'test_gal'
}

########################################################################

class TestGalaxyFinder( unittest.TestCase ):

  def setUp( self ):

    # Get input data
    comoving_halo_coords = np.array([ [ 29414.96458784,  30856.75007114,  32325.90901812],
                                      [ 31926.42103071,  51444.46756529,   1970.1967437 ] ])

    self.redshift = gal_finder_data_p['redshift']
    self.hubble = gal_finder_data_p['hubble']
    halo_coords = comoving_halo_coords/(1. + self.redshift)/self.hubble

    # Make the necessary data_p
    self.data_p = gal_finder_data_p

    self.galaxy_finder = galaxy_finding.GalaxyFinder( halo_coords, self.data_p ) 

    # Get the necessary reader.
    self.galaxy_finder.ahf_reader = ahf_reading.AHFReader( self.data_p['sdir'] )
    
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
    }

    # Do the actual calculation
    galaxy_finder = galaxy_finding.GalaxyFinder( particle_positions, self.data_p )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
      npt.assert_allclose( expected[key], actual[key] )

########################################################################

class TestParticleTrackGalaxyFinder( unittest.TestCase ):

  def setUp( self ):

    self.originalfile = './tests/test_data/tracking_output/ptrack_test.hdf5'
    self.savefile = './tests/test_data/tracking_output/ptrack_test_gal.hdf5'

    # Make the base particle tracking file
    os.system( 'cp {} {}'.format( self.originalfile, self.savefile ) )

  ########################################################################

  def tearDown( self ):

    os.system( 'rm {}'.format( self.savefile ) )

  ########################################################################

  def test_find_galaxies_for_particle_tracks( self ):

    particle_track_gal_finder = galaxy_finding.ParticleTrackGalaxyFinder( ptrack_gal_finder_data_p )
    particle_track_gal_finder.find_galaxies_for_particle_tracks()

    f = h5py.File( self.savefile, 'r' )

    # What we expect (calculated using the positions of the particles in snum 500 )
    particle_positions = np.array([
       [ 34463.765625  ,  37409.45703125,  38694.92578125],
       [ 35956.61328125,  38051.79296875,  39601.91796875],
       [ 34972.05078125,  39093.890625  ,  39445.94921875],
       [ 35722.984375  ,  38222.14453125,  39245.52734375],
    ])
    galaxy_finder = galaxy_finding.GalaxyFinder( particle_positions, gal_finder_data_p )
    expected = galaxy_finder.find_ids()

    for key in expected.keys():
      npt.assert_allclose( expected[key], f[key][...][:,-1] )

    # Make sure we still have the original data...
    assert 'rho' in f.keys()

    f.close()
