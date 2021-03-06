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

import linefinder.analyze_data.events as analyze_events

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestEventsStartup( unittest.TestCase ):

  def test_init( self ):

    events = analyze_events.Events( tracking_dir, tag )

    assert events.parameters['tag'] == tag

