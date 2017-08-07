#!/usr/bin/env python
'''Testing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
import mock
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import worldline.analyze_data.analyze_worldline_set as analyze_worldline_set

########################################################################
# Commonly useful input variables

defaults = {
  'data_dir' : './tests/data/tracking_output_for_analysis',
  'tag' : 'analyze',
}

########################################################################

class TestWorldlineSetStartUp( unittest.TestCase ):

  @mock.patch( 'worldline.analyze_data.analyze_worldlines.Worldlines.__init__' )
  def test_init( self, mock_constructor ):

    variations = {
      'a' : { 'data_dir' : 'data_dir_a' },
      'b' : { 'data_dir' : 'data_dir_b' },
      'c' : { 'tag' : 'tag_c' },
      'd' : {},
    }

    mock_constructor.side_effect = [ None, ]*len( variations )

    calls = [
      mock.call( data_dir='data_dir_a', tag=defaults['tag'] ),
      mock.call( data_dir='data_dir_b', tag=defaults['tag'] ),
      mock.call( data_dir=defaults['data_dir'], tag='tag_c' ),
      mock.call( data_dir=defaults['data_dir'], tag=defaults['tag'] ),
    ]

    worldline_set = analyze_worldline_set.WorldlineSet( defaults, variations )

    mock_constructor.assert_has_calls( calls, any_order=True )

    # Check that it behaves like a dict
    assert len( worldline_set ) == len( variations )
    for key in worldline_set.keys():
      assert key in variations
    for item in worldline_set:
      assert item in variations

