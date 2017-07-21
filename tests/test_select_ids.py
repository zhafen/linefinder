#!/usr/bin/env python
'''Testing for select_ids.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
from mock import call, patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import unittest

from worldline import select_ids

########################################################################

# For IDSelector
default_kwargs = {
  'snum_start' : 500,
  'snum_end' : 600,
  'snum_step' : 100,
  'ptypes' : [0, 4],

  'sdir' : './tests/data/stars_included_test_data',
  'load_additional_ids' : True,
  'ahf_index' : 600,
  'analysis_dir' : './tests/data/ahf_test_data',
}

# For SnapshotIDSelector
default_snap_kwargs = {
  'sdir' : './tests/data/stars_included_test_data',
  'snum' : 500,
  'ptype' : 0,
  'load_additional_ids' : False,
  'ahf_index' : 600,
  'analysis_dir' : './tests/data/ahf_test_data',
}

newids_snap_kwargs = copy.deepcopy( default_snap_kwargs )
newids_snap_kwargs['load_additional_ids'] = True

default_data_filters = [
  { 'data_key' : 'Rf', 'data_min' : 0., 'data_max' : 1., },
  { 'data_key' : 'T', 'data_min' : 1e4, 'data_max' : 1e6, },
]

########################################################################
########################################################################

class TestSnapshotIDSelector( unittest.TestCase ):

  def setUp( self ):

    self.snapshot_id_selector = select_ids.SnapshotIDSelector( **default_snap_kwargs )

    # Setup some test data with a range of values useful to us.
    self.snapshot_id_selector.data['R'] = np.array( [ 0.5, 1.2, 0.75, 0.1, 0.3, 1.5 ] )*self.snapshot_id_selector.length_scale
    self.snapshot_id_selector.data['T'] = np.array( [ 1e2, 1.1e4, 1e7, 1e5, 0.5e6, 0.5e5 ] )

  ########################################################################

  def test_default( self ):

    expected = 0.16946
    actual = self.snapshot_id_selector.redshift
    npt.assert_allclose( expected, actual, atol=1e-5 )

  ########################################################################

  def test_filter_data( self ):

    # Expected result from applying the default filters
    expected_dict = {
      'Rf' : np.array( [ 0, 1, 0, 0, 0, 1 ] ).astype( bool ),
      'T' : np.array( [ 1, 0, 1, 0, 0, 0 ] ).astype( bool ),
    }

    # Apply the filters.
    self.snapshot_id_selector.filter_data( default_data_filters )

    masks = self.snapshot_id_selector.data_masker.masks
    assert len(masks) == 2

    # Check the results
    for mask in masks:
      expected = expected_dict[mask['data_key']]
      actual = mask['mask']
      npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_ids( self ):

    # Make masks (easiest just to use the function I just tested, even if it's not perfect unit testing....)
    self.snapshot_id_selector.filter_data( default_data_filters )

    expected = np.array( [ 10952235, 36091289, ] )
    actual = self.snapshot_id_selector.get_ids()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_format_ids( self ):

    expected = set( [ 10952235, 36091289, ] )

    actual = self.snapshot_id_selector.format_ids( np.array( [ 10952235, 36091289, ] ) )

    assert expected == actual

  ########################################################################

  def test_select_ids_snapshot( self ):

    expected = set( [ 10952235, 36091289, ] )

    actual = self.snapshot_id_selector.select_ids_snapshot( default_data_filters )

    assert expected == actual

########################################################################
########################################################################

class TestWithChildIDs( unittest.TestCase ):

  def setUp( self ):

    self.snapshot_id_selector = select_ids.SnapshotIDSelector( **newids_snap_kwargs )

    # Setup some test data with a range of values useful to us.
    self.snapshot_id_selector.data['R'] = np.array( [ 0.5, 1.2, 0.75, 0.1, 0.3, 1.5 ] )*self.snapshot_id_selector.length_scale
    self.snapshot_id_selector.data['T'] = np.array( [ 1e2, 1.1e4, 1e7, 1e5, 0.5e6, 0.5e5 ] )

    self.selected_ids = ( np.array( [ 10952235, 36091289, ] ), np.array( [ 0, 893109954, ] ) )

    self.ids_set = set( [ (10952235, 0), (36091289, 893109954) ] )

  ########################################################################

  def test_get_ids( self ):

    # Make masks
    self.snapshot_id_selector.filter_data( default_data_filters )

    expected = self.selected_ids

    actual = self.snapshot_id_selector.get_ids()

    for i in range(2):
      npt.assert_allclose( expected[i], actual[i] )
    
  ########################################################################

  def test_format_ids( self ):

    expected = self.ids_set

    actual = self.snapshot_id_selector.format_ids( self.selected_ids )

    assert expected == actual

  ########################################################################

  def test_select_ids_snapshot( self ):

    expected = self.ids_set

    actual = self.snapshot_id_selector.select_ids_snapshot( default_data_filters )

    assert expected == actual

########################################################################
########################################################################

class TestIDSelector( unittest.TestCase ):

  def setUp( self ):

    self.side_effects = [
      set( [ (10952235, 0), (36091289, 893109954) ] ),
      set( [ (10952235, 0), (123456, 35) ] ),
      set( [ (1573, 0), (12, 35) ] ),
      set( [ (15, 4), (0, 0) ] ),
    ]

    self.id_selector = select_ids.IDSelector( **default_kwargs )

  ########################################################################

  @patch( 'worldline.select_ids.SnapshotIDSelector.__init__' )
  @patch( 'worldline.select_ids.SnapshotIDSelector.select_ids_snapshot' )
  def test_get_selected_ids( self, mock_select_ids_snapshot, mock_constructor, ):

    # Mock setup
    mock_constructor.side_effect = [ None, ]*4
    mock_select_ids_snapshot.side_effect = self.side_effects

    call_kwargs = [ copy.deepcopy( newids_snap_kwargs ) for i in range(4) ]
    call_kwargs[0]['snum'] = 500
    call_kwargs[1]['snum'] = 500
    call_kwargs[2]['snum'] = 600
    call_kwargs[3]['snum'] = 600
    call_kwargs[0]['ptype'] = 0
    call_kwargs[1]['ptype'] = 4
    call_kwargs[2]['ptype'] = 0
    call_kwargs[3]['ptype'] = 4
    calls = [ call( **call_kwarg ) for call_kwarg in call_kwargs ]

    # Actually run the thing
    actual = self.id_selector.get_selected_ids( default_data_filters )
    expected = set( [ (10952235, 0), (36091289, 893109954), (123456, 35), (1573, 0), (12, 35), (15, 4), (0, 0) ])
    assert expected == actual

    mock_constructor.assert_has_calls( calls )
