#!/usr/bin/env python
'''Testing.

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

import worldline.analyze_data.analyze_worldlines as analyze_worldlines

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestWorldlines( unittest.TestCase ):

  def setUp( self ):

    self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag )

  ########################################################################

  def test_load_ptracks( self ):

    assert self.worldlines.ptracks.parameters['tag'] == tag

  ########################################################################

  def test_load_galids( self ):

    assert self.worldlines.galids.parameters['tag'] == tag

  ########################################################################

  def test_load_classifications( self ):

    assert self.worldlines.classifications.parameters['tag'] == tag
