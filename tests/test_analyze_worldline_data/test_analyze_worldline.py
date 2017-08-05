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

import worldline.analyze_worldline_data.analyze_worldline as analyze_worldline

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestWorldlineData( unittest.TestCase ):

  def setUp( self ):

    self.w_data = analyze_worldline.WorldlineData( tracking_dir, tag )

  ########################################################################

  def test_load_ptrack_data( self ):

    assert self.w_data.ptrack_data.parameters['tag'] == tag

  ########################################################################

  def test_load_galfind_data( self ):

    assert self.w_data.galfind_data.parameters['tag'] == tag

  ########################################################################

  def test_load_classified_data( self ):

    assert self.w_data.classified_data.parameters['tag'] == tag
