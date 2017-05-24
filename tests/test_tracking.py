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

    id_finder = tracking.IDFinder()

    # The name of the function.
    self.fn = id_finder.select_ids

    # Dummy data set
    id_finder.target_ids = np.array([ 0, 2 ])
    id_finder.full_snap_data = {
      'id' : np.array([ 0, 1, 2 ]),
      'rho' : np.array([ 1.5, 1., 2. ]),
    }

  ########################################################################

  def test_runs(self):
    '''Test that it even runs'''

    self.fn()

    assert True
