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
from particle_tracking import galaxy_finding

########################################################################

class TestGalaxyFinder( unittest.TestCase ):

  def setUp( self ):

    # Get input data
    comoving_halo_coords = np.array([ [ 29414.96458784,  30856.75007114,  32325.90901812],
                                      [ 31926.42103071,  51444.46756529,   1970.1967437 ] ])

    self.redshift = 0.16946003
    self.hubble_param = 0.70199999999999996
    halo_coords = comoving_halo_coords/(1. + self.redshift)/self.hubble_param

    self.galaxy_finder = galaxy_finding.GalaxyFinder( halo_coords, self.redshift, self.hubble_param )

    self.galaxy_finder.ahf_reader = ahf_reading.AHFReader( './tests/test_data/ahf_test_data' )
    
    self.galaxy_finder.ahf_reader.get_ahf_halos( 500 )
    self.galaxy_finder.ahf_reader.get_ahf_mtree_idx( 500 )

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
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble_param

    self.galaxy_finder.n_particles = 3

    expected = np.array( [0, 6962, 7] )
    actual = self.galaxy_finder.find_smallest_containing_halo()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_smallest_host_halo_none( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      ])

    expected = np.array( [999999, 999999] )
    actual = self.galaxy_finder.find_smallest_containing_halo()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_host_halo( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812],
      [ 29058.92308963,  31944.12724155,  32312.25393713],
      [ 28213.25906375,  30682.61827584,  31675.12721302],
      [ 29467.07226789,  30788.6179313 ,  32371.38749237],
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble_param

    self.galaxy_finder.n_particles = 4

    expected = np.array( [0, 3, 10, 0] )
    actual = self.galaxy_finder.find_host_halo()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_host_halo_none( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      ])

    expected = np.array( [999999, 999999] )
    actual = self.galaxy_finder.find_smallest_containing_halo()

    npt.assert_allclose( expected, actual )
