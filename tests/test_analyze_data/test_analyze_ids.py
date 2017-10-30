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

import pathfinder.analyze_data.ids as analyze_ids

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestGalIDsStartup( unittest.TestCase ):

  def test_init( self ):

    ids = analyze_ids.IDs( tracking_dir, tag )

    assert ids.parameters['tag'] == tag

