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

import linefinder.analyze_data.classifications as analyze_classifications

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

########################################################################

class TestClassificationsStartup( unittest.TestCase ):

  def test_init( self ):

    classifications = analyze_classifications.Classifications( tracking_dir, tag )

    assert classifications.parameters['tag'] == tag

########################################################################

class TestClassifications( unittest.TestCase ):

  def setUp( self ):

    self.classifications = analyze_classifications.Classifications( tracking_dir, tag )

  ########################################################################

  def test_get_data( self ):

    actual = self.classifications.get_data( 'is_pristine' )
  
    expected = self.classifications.data['is_pristine']

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_data_mask( self ):

    mask = np.array( [ 1, 0, 1, 0, ] ).astype( bool )
    actual = self.classifications.get_data( 'is_mass_transfer', mask=mask )

    expected = np.array( [ 1, 0, ] ).astype( bool )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_data_slice( self ):
    
    mask = np.array( [ 1, 0, 0, 1, ] ).astype( bool )
    actual = self.classifications.get_data( 'is_wind', mask=mask, slice_index=1 )

    expected = np.array( [ 1, 1, ] ).astype( bool )

    npt.assert_allclose( expected, actual )

