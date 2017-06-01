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

  def test_get_ahf_halos( self ):

    self.ahf_reader.get_ahf_halos( 600 )

    expected = 792
    actual = self.ahf_reader.ahf_halos['numSubStruct'][0]
    npt.assert_allclose( expected, actual )

