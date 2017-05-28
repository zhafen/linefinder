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
from particle_tracking import readsnap

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

class TestConcatenateParticleData( unittest.TestCase ):

  def setUp( self ):

    self.id_finder = tracking.IDFinder()

    # The name of the function.
    self.fn = self.id_finder.concatenate_particle_data

  ########################################################################

  def test_basic( self ):
    '''Basically, does it work?'''

    # Input
    self.id_finder.sdir = './tests/test_data/test_data_with_new_id_scheme'
    self.id_finder.snum = 600
    self.id_finder.types = [0,]
    self.id_finder.target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
    self.id_finder.target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])

    actual = self.id_finder.concatenate_particle_data()

    expected = {
      'id' : np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235]),
      'child_id' : np.array([1945060136, 0, 0, 938428052, 893109954, 0]),
      }

    for key in expected.keys():
      npt.assert_allclose( actual[key], expected[key] )

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
      'child_id' : np.array([1945060136, 0, 0, 938428052, 893109954, 0]),
      'rho' : np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
    }

    dfid = self.fn()

    expected = {
      'id' : self.id_finder.target_ids,
      'child_id' : self.id_finder.target_child_ids,
      'rho' : np.array([ 3.45104532e-08, 3.80374093e-10, 6.80917722e-09, 1.54667816e-08 ]),
    }

    for key in dfid.keys():
      npt.assert_allclose( dfid[key], expected[key] )

########################################################################

class TestFindIds( unittest.TestCase ):

  def setUp( self ):

    self.id_finder = tracking.IDFinder()

    # The name of the function.
    self.fn = self.id_finder.find_ids

  ########################################################################

  def test_basic( self ):
    '''Basically, does it work?.'''

    # Input
    sdir = './tests/test_data/test_data_with_new_id_scheme'
    snum = 600
    types = [0,]
    target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
    target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])

    # My knowledge, by hand
    target_inds = [4, 0, 1, 5]
    P = readsnap.readsnap( sdir, snum, 0, True, cosmological=True )

    expected = {
      'id' : target_ids,
      'child_id' : target_child_ids,
      'rho' : np.array([ P['rho'][ind] for ind in target_inds ]),
    }

    dfid, redshift = self.fn( sdir, snum, types, target_ids, \
                              target_child_ids=target_child_ids)

    for key in expected.keys():
      npt.assert_allclose( dfid[key], expected[key] )

    # Make sure the redshift's right too
    npt.assert_allclose( redshift, 0. )

########################################################################

class TestSaveTargetedParticles( unittest.TestCase ):

  def setUp( self ):

    self.id_finder_full = tracking.IDFinderFull()

    # The name of the function.
    self.fn = self.id_finder_full.save_targeted_particles

  ########################################################################

  def test_basic( self ):
    '''Basically, does it work?'''

    assert False

########################################################################

# TODO
#class TestReadAHFParticles( unittest.TestCase ):
#
#  def setUp( self ):
#
#    # The name of the function.
#    self.fn = tracking.read_ahf_particles
#
#  ########################################################################
#
#  def test_runs( self ):
#    '''Test if it even runs.'''
#
#    assert False
