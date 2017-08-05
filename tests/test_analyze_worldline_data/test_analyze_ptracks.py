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

import worldline.analyze_data.analyze_ptracks as analyze_ptracks

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestPTracksStartup( unittest.TestCase ):

  def test_init( self ):

    ptracks = analyze_ptracks.PTracks( tracking_dir, tag )

    assert ptracks.parameters['tag'] == tag

