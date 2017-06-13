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

sdir = './tests/test_data/ahf_test_data'

########################################################################

class TestAHFReader( unittest.TestCase ):

  def setUp( self ):

    self.ahf_reader = ahf_reading.AHFReader( sdir )

  ########################################################################

  def test_get_halos( self ):

    self.ahf_reader.get_halos( 500 )

    expected = 789
    actual = self.ahf_reader.ahf_halos['numSubStruct'][0]
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_idx( self ):

    self.ahf_reader.get_mtree_idx( 500 )

    expected = 10
    actual = self.ahf_reader.ahf_mtree_idx['HaloID(1)'][10]
    npt.assert_allclose( expected, actual )

    expected = 11
    actual = self.ahf_reader.ahf_mtree_idx['HaloID(2)'][10]
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_halo_files( self ):

    self.ahf_reader.get_mtree_halos( 'snum' )

    # Halo mass at z=0 for mtree_halo_id = 0
    expected = 7.5329e+11
    actual = self.ahf_reader.mtree_halos[0]['Mvir'][600]
    npt.assert_allclose( expected, actual )

    # ID at an early snapshot (snap 10) for halo file 0
    expected = 10
    actual = self.ahf_reader.mtree_halos[0]['ID'][10]
    npt.assert_allclose( expected, actual )

    # ID at snapshot 30 for halo file 2
    expected = 60
    actual = self.ahf_reader.mtree_halos[2]['ID'][30]
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_halo_quantity( self ):

    self.ahf_reader.get_mtree_halos( 'snum' )

    actual = self.ahf_reader.get_mtree_halo_quantity( 'ID', 600, 'snum' )
    expected = np.arange( 0, 20 )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mtree_halo_id_matches( self ):
    '''Test that the ID in the mtree_halo_id is exactly what we expect it to be.'''

    self.ahf_reader.get_mtree_halos( 'snum' )
    halo_id = self.ahf_reader.mtree_halos[10]['ID'][500]

    # First make sure we have the right ID
    assert halo_id == 11 # just looked this up manually.

    # Now make sure we have the right x position, as a check
    expected = 28213.25906375 # Halo 11 X position at snum 500

    self.ahf_reader.get_halos( 500 )
    actual = self.ahf_reader.ahf_halos['Xc'][ halo_id ]

    npt.assert_allclose( expected, actual )

########################################################################

  def test_get_pos_or_vel( self ):

    # Snap 550, mt halo 0, position
    expected = np.array([ 29372.26565053,  30929.16894187,  32415.81701217])

    # Get the values
    self.ahf_reader.get_mtree_halos( 'snum' )
    mt_halo_0_pos = self.ahf_reader.get_pos_or_vel( 'pos', 0, 550 )
    actual = mt_halo_0_pos
    
    npt.assert_allclose( expected, actual )
