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

kwargs = {
  'ahf_data_dir' : './tests/data/ahf_test_data',
  'ahf_index' : 600,
}

########################################################################

class TestWorldlines( unittest.TestCase ):

  def setUp( self ):

    self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

  ########################################################################

  def test_load_ptracks( self ):

    assert self.worldlines.ptracks.parameters['tag'] == tag

  ########################################################################

  def test_load_galids( self ):

    assert self.worldlines.galids.parameters['tag'] == tag

  ########################################################################

  def test_load_classifications( self ):

    assert self.worldlines.classifications.parameters['tag'] == tag

  ########################################################################

  def test_get_parameters( self ):

    data_types = [ 'ptracks', 'galids', 'classifications' ]
    expected = {}
    for data_type in data_types:
      filename = '{}_analyze.hdf5'.format( data_type )
      f = h5py.File( os.path.join( tracking_dir, filename ), 'r' )
      expected[data_type] = f['parameters']

    actual = self.worldlines.get_parameters()

    for data_key in actual.keys():
      for key in actual[data_key].keys():
        if not isinstance( actual[data_key][key], np.ndarray ):
          self.assertEqual( actual[data_key][key], expected[data_key].attrs[key] )

########################################################################

class TestWorldlineDataMasker( unittest.TestCase ):

  def setUp( self ):

    self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

  ########################################################################

  def test_given_mask( self ):

    mask = np.array( [
      [ 1, 1, 0, ],
      [ 0, 0, 0, ],
      [ 1, 1, 1, ],
      [ 1, 1, 1, ],
    ] ).astype( bool )

    actual = self.worldlines.data_masker.get_masked_data( 'T', mask )
    expected = np.array( [ 58051.05859375, 12212.33984375,  812602.1875    ,   25435.59375 ])
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_given_mask_slice( self ):

    mask = np.array( [ True, False, True, True ] )

    actual = self.worldlines.data_masker.get_masked_data( 'T', mask, ( slice(None), 1 ) )
    expected = np.array( [ 812602.1875 ])
    npt.assert_allclose( expected, actual )
