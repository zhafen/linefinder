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

import pathfinder.analyze_data.galids as analyze_galids

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestGalIDsStartup( unittest.TestCase ):

  def test_init( self ):

    galids = analyze_galids.GalIDs( tracking_dir, tag )

    assert galids.parameters['tag'] == tag

