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

default_kwargs = {
  'sdir' : './tests/data/stars_included_test_data',
  'snum' : 500,
  'ptype' : 0,
  'load_additional_ids' : True,
  'ahf_index' : 600,
  'analysis_dir' : './tests/data/ahf_test_data',
}

########################################################################
########################################################################

class TestSnapshotIDSelector( unittest.TestCase ):

  def setUp( self ):

    self.snapshot_id_selector = select_ids.SnapshotIDSelector( **default_kwargs )

  ########################################################################

  def test_default( self ):

    expected = 0.16946
    actual = self.snapshot_id_selector.p_data.redshift
    npt.assert_allclose( expected, actual, atol=1e-5 )

########################################################################
########################################################################

class TestWithChildIDs( unittest.TestCase ):

  def test_select_ids( self ):
    assert False, "Need to do."
