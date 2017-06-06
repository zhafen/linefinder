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


    self.ahf_reader.get_mtree_halos()

    # Halo mass at z=0 for mtree_halo_id = 0
    expected = 7.5329e+11
    actual = self.ahf_reader.mtree_halos[0]['Mvir'][599]
    npt.assert_allclose( expected, actual )

    # ID at the second snapshot for halo_file 0
    expected = 8
    actual = self.ahf_reader.mtree_halos[0]['ID'][1]
    npt.assert_allclose( expected, actual )
