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

import worldline.analyze_worldline_data.classified as classified

########################################################################
# Commonly useful input variables

tracking_dir = './tests/test_data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestClassifiedDataStartup( unittest.TestCase ):

  def test_init( self ):

    classified_data = classified.ClassifiedData( tracking_dir, tag )

    assert classified_data.data_attrs['tag'] == tag

    
