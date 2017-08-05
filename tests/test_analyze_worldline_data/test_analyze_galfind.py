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

import worldline.analyze_worldline_data.analyze_galfind as analyze_galfind

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestGalfindDataStartup( unittest.TestCase ):

  def test_init( self ):

    galfind_data = analyze_galfind.GalfindData( tracking_dir, tag )

    assert galfind_data.parameters['tag'] == tag

