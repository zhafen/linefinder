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

import worldline.analyze_worldline_data.analyze_classified as analyze_classified

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestClassifiedDataStartup( unittest.TestCase ):

  def test_init( self ):

    classified_data = analyze_classified.ClassifiedData( tracking_dir, tag )

    assert classified_data.parameters['tag'] == tag

########################################################################

class TestClassifiedData( unittest.TestCase ):

  def setUp( self ):

    self.classified_data = analyze_classified.ClassifiedData( tracking_dir, tag )

  ########################################################################

  def test_get_data( self ):

    actual = self.classified_data.get_data( 'is_pristine' )
  
    expected = self.classified_data.data['is_pristine']

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_data_mask( self ):

    mask = np.array( [ 1, 0, 1, 0, ] ).astype( bool )
    actual = self.classified_data.get_data( 'is_mass_transfer', mask=mask )

    expected = np.array( [ 1, 0, ] ).astype( bool )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_data_slice( self ):
    
    mask = np.array( [ 1, 0, 0, 1, ] ).astype( bool )
    actual = self.classified_data.get_data( 'is_wind', mask=mask, slice_index=1 )

    expected = np.array( [ 1, 1, ] ).astype( bool )

    npt.assert_allclose( expected, actual )

########################################################################

class TestCalcBaseFractions( unittest.TestCase ):

  def setUp( self ):

    self.classified_data = analyze_classified.ClassifiedData( tracking_dir, tag )

  ########################################################################

  def test_calc_base_fractions( self ):

    actual = self.classified_data.calc_base_fractions()

    expected = {
      'fresh accretion' : 0.25,
      'merger' : 0.5,
      'intergalactic transfer' : 0.25,
      'wind' : 0.5,
    }

    for key in actual.keys():
      npt.assert_allclose( expected[key], actual[key] )

  ########################################################################

  def test_calc_base_fractions_mask( self ):

    mask = np.array( [ 1, 0, 1, 0, ] ).astype( bool )

    actual = self.classified_data.calc_base_fractions( mask=mask )

    expected = {
      'fresh accretion' : 0.,
      'merger' : 0.5,
      'intergalactic transfer' : 0.5,
      'wind' : 0.5,
    }

    for key in actual.keys():
      npt.assert_allclose( expected[key], actual[key] )

