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

import worldline.analyze_worldline_data.analyze_ptrack as analyze_ptrack

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestPtrackDataStartup( unittest.TestCase ):

  def test_init( self ):

    ptrack_data = analyze_ptrack.PtrackData( tracking_dir, tag )

    assert ptrack_data.parameters['tag'] == tag

