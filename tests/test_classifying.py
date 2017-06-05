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

from particle_tracking import classifying

default_data_p = {
  'trackdir' : './tests/test_data/tracking_output',
  'tag' : 'test',
}

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

class TestIdentifyAccrectionEjectionAndMergers( unittest.TestCase ):

  def setUp( self ):

    self.classifier = classifying.Classifier( default_data_p )

  ########################################################################

  def test_basic():

    assert False
