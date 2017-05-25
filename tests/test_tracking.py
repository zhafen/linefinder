#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import numpy.testing as npt
import pdb
import unittest

from particle_tracking import tracking

########################################################################

class SaneEqualityArray(np.ndarray):
  '''Numpy array subclass that allows you to test if two arrays are equal.'''

  def __eq__(self, other):
      return (isinstance(other, np.ndarray) and self.shape == other.shape and np.allclose(self, other))

def sane_eq_array(list_in):
  '''Wrapper for SaneEqualityArray, that takes in a list.'''

  arr = np.array(list_in)

  return arr.view(SaneEqualityArray)

########################################################################

class TestSelectIDs(unittest.TestCase):

  def setUp(self):

    self.id_finder = tracking.IDFinder()

    # The name of the function.
    self.fn = self.id_finder.select_ids

  ########################################################################

  def test_runs(self):
    '''Test that it even runs'''

    # Dummy data set
    self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
    self.id_finder.full_snap_data = {
      'id' : np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
      'rho' : np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
    }

    self.fn()

    assert True

  ########################################################################

  def test_works_simple( self ):
    '''Test that it runs in a simple case where there's no issues with duplicates.'''

    # Dummy data set
    self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
    self.id_finder.full_snap_data = {
      'id' : np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
      'rho' : np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
    }

    dfid = self.fn()

    expected = {
      'id' : np.array([ 38913508, 3211791, 10952235 ]),
      'rho' : np.array([ 3.45104532e-08,  6.80917722e-09, 1.54667816e-08 ]),
    }

    for key in dfid.keys():
      npt.assert_allclose( dfid[key], expected[key] )

  ########################################################################

  def test_works_duplicates( self ):

    # Dummy data set
    self.id_finder.target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
    self.id_finder.target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])
    self.id_finder.full_snap_data = {
      'id' : np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235]),
      'rho' : np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
    }

    dfid = self.fn()

    expected = {
      'id' : self.id_finder.target_ids,
      'rho' : np.array([ 3.45104532e-08, 3.80374093e-10, 6.80917722e-09, 1.54667816e-08 ]),
    }

    for key in dfid.keys():
      npt.assert_allclose( dfid[key], expected[key] )

