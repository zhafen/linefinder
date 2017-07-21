#!/usr/bin/env python
'''Testing for select_ids.py

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
import unittest

from worldline import select_ids

########################################################################

default_snap_kwargs = {
  'sdir' : './tests/data/stars_included_test_data',
  'snum' : 500,
  'ptype' : 0,
  'load_additional_ids' : True,
  'ahf_index' : 600,
  'analysis_dir' : './tests/data/ahf_test_data',
}

default_data_filters = [
  { 'data_key' : 'Rf', 'data_min' : 0., 'data_max' : 1., },
  { 'data_key' : 'T', 'data_min' : 1e4, 'data_max' : 1e6, },
]

########################################################################
########################################################################

class TestSnapshotIDSelector( unittest.TestCase ):

  def setUp( self ):

    self.snapshot_id_selector = select_ids.SnapshotIDSelector( **default_snap_kwargs )

  ########################################################################

  def test_default( self ):

    expected = 0.16946
    actual = self.snapshot_id_selector.p_data.redshift
    npt.assert_allclose( expected, actual, atol=1e-5 )

  ########################################################################

  def test_filter_data( self ):

    # Setup some test data with a range of values useful to us.
    self.snapshot_id_selector.p_data.data['R'] = np.array( [ 0.5, 1.2, 0.75, 0.1, 0.3, 1.5 ] )*self.snapshot_id_selector.p_data.length_scale
    self.snapshot_id_selector.p_data.data['T'] = np.array( [ 1e2, 1.1e4, 1e7, 1e5, 0.5e6, 0.5e5 ] )

    # Expected result from applying the default filters
    expected_dict = {
      'Rf' : np.array( [ 0, 1, 0, 0, 0, 1 ] ).astype( bool ),
      'T' : np.array( [ 1, 0, 1, 0, 0, 0 ] ).astype( bool ),
    }

    # Apply the filters.
    self.snapshot_id_selector.filter_data( default_data_filters )

    masks = self.snapshot_id_selector.p_data.data_masker.masks
    assert len(masks) == 2

    # Check the results
    for mask in masks:
      expected = expected_dict[mask['data_key']]
      actual = mask['mask']
      npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestWithChildIDs( unittest.TestCase ):

  def test_select_ids( self ):
    assert False, "Need to do."
