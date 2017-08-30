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

  def test_load_ids( self ):

    assert self.worldlines.ids.parameters['tag'] == tag

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

  def test_load_events( self ):

    assert self.worldlines.events.parameters['tag'] == tag

  ########################################################################

  def test_get_parameters( self ):

    data_types = [ 'ids', 'ptracks', 'galids', 'classifications' ]
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

class TestWorldlinesDifferentTags( unittest.TestCase ):

  def setUp( self ):

    self.kwargs = {
      'data_dir' : './tests/data/tracking_output_for_analysis',
      'tag' : 'analyze',

      'ptracks_tag' : 'alt_tag1',
      'galids_tag' : 'alt_tag2',
      'classifications_tag' : 'alt_tag3',

      'ahf_data_dir' : './tests/data/ahf_test_data',
      'ahf_index' : 600,
    }
        
    self.worldlines = analyze_worldlines.Worldlines( **self.kwargs )

  ########################################################################

  @mock.patch( 'worldline.analyze_data.analyze_classifications.Classifications.__init__' )
  @mock.patch( 'worldline.analyze_data.analyze_galids.GalIDs.__init__' )
  @mock.patch( 'worldline.analyze_data.analyze_ptracks.PTracks.__init__' )
  def test_different_tags( self, mock_ptracks, mock_galids, mock_classifications ):

    mock_ptracks.side_effect = [ None, ]
    mock_galids.side_effect = [ None, ]
    mock_classifications.side_effect = [ None, ]
    
    self.worldlines.ptracks
    ptrack_kwargs = {
      'ahf_data_dir' : './tests/data/ahf_test_data',
      'ahf_index' : 600,
    }
    mock_ptracks.assert_called_with(
      './tests/data/tracking_output_for_analysis',
      'alt_tag1',
      store_ahf_reader=True,
      **ptrack_kwargs )

    self.worldlines.galids
    mock_galids.assert_called_with(
      './tests/data/tracking_output_for_analysis',
      'alt_tag2' )

    self.worldlines.classifications
    mock_classifications.assert_called_with(
      './tests/data/tracking_output_for_analysis',
      'alt_tag3' )

########################################################################
########################################################################

class TestWorldlineGetData( unittest.TestCase ):

  def setUp( self ):

    self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

  ########################################################################

  @mock.patch( 'galaxy_diver.analyze_data.simulation_data.SimulationData.get_data' )
  def test_basic( self, mock_get_data ):

    self.worldlines.get_data( 'Rx' )

    mock_get_data.assert_called_once_with( 'Rx', sl=None )

########################################################################
########################################################################

class TestWorldlineCalcData( unittest.TestCase ):

  def setUp( self ):

    self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

  ########################################################################

  def test_calc_fresh_accretion( self ):

    # Setup test data
    self.worldlines.data['is_pristine'] = np.array( [ 1, 1, 1, 0, ] ).astype( bool )
    self.worldlines.data['is_wind'] = np.array([
      [ 1, 1, 0, ],
      [ 1, 0, 0, ],
      [ 0, 0, 0, ],
      [ 1, 1, 0, ],
    ]).astype( bool )

    self.worldlines.calc_fresh_accretion()

    actual = self.worldlines.data['is_fresh_accretion']
    expected = np.array([
      [ 0, 0, 1, ],
      [ 0, 1, 1, ],
      [ 1, 1, 1, ],
      [ 0, 0, 0, ],
    ]).astype( bool )

    npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestWorldlineProperties( unittest.TestCase ):

  def setUp( self ):

    self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

  ########################################################################

  def test_n_particles_snapshot( self ):

    actual = self.worldlines.n_particles_snapshot
    expected = 12
    self.assertEqual( expected, actual )

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

    actual = self.worldlines.data_masker.get_masked_data( 'T', mask=mask )
    expected = np.array( [ 58051.05859375, 12212.33984375,  812602.1875    ,   25435.59375 ])
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_given_mask_slice( self ):

    mask = np.array( [ True, False, True, True ] )

    actual = self.worldlines.data_masker.get_masked_data(
      'T',
      mask=mask,
      sl=( slice(None), 1 ),
      apply_slice_to_mask=False,
      preserve_mask_shape=True,
    )
    expected = np.array( [ 812602.1875 ])
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data_classification( self ):

    actual = self.worldlines.get_masked_data( 'T',
      classification='is_mass_transfer',
      sl=( slice(None), slice(0,2)  ),
      )
    expected = np.array( [ 12212.33984375, 812602.1875 ])
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data_classification_combined( self ):

    mask = np.array( [
      [ 0, 0, 0, ],
      [ 0, 0, 0, ],
      [ 0, 1, 1, ],
      [ 1, 1, 1, ],
    ] ).astype( bool )

    actual = self.worldlines.data_masker.get_masked_data( 'T', mask = mask,  classification='is_preprocessed', sl=( slice(None), slice(0,2), ) )
    expected = np.array( [ 12212.33984375,   812602.1875, 42283.62890625, ] )
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data_before_first_acc( self ):

    self.worldlines.events.data['redshift_first_acc'] = np.array([ 30., -1., 0., 30. ])

    actual = self.worldlines.data_masker.get_masked_data( 'T', mask_after_first_acc=True,  classification='is_preprocessed', sl=( slice(None), slice(0,2), ) )
    expected = np.array( [ 12212.33984375,   812602.1875, 4107.27490234, ] )
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data_after_first_acc( self ):

    self.worldlines.events.data['redshift_first_acc'] = np.array([ 30., -1., 0., 30. ])

    actual = self.worldlines.data_masker.get_masked_data( 'T', mask_before_first_acc=True,  classification='is_preprocessed', sl=( slice(None), slice(0,2), ) )
    expected = np.array( [ 42283.62890625,  20401.44335938,  115423.2109375, ] )
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mask_default_and_other( self ):

    self.worldlines.data_masker.masks.append(
      {
        'data_key' : 'NA',
        'data_min' : np.nan,
        'data_max' : np.nan,
        'mask' : np.array([
          [ 1, 1, 0, ],
          [ 1, 0, 0, ],
          [ 1, 0, 1, ],
          [ 0, 0, 0, ],
        ]),
      }
    )
  
    actual = self.worldlines.data_masker.get_mask()
    expected = self.worldlines.data_masker.masks[0]['mask']

    npt.assert_allclose( expected, actual )
    
  ########################################################################

  def test_get_masked_data_defaults( self ):

    actual = self.worldlines.get_masked_data( 'T' )
    expected = np.array([
      [  22864.45898438,  379941.71875   ,   58051.05859375],
      [  12212.33984375,  812602.1875    ,   25435.59375   ],
      [  42283.62890625,    4107.27490234,   10226.62792969],
      [  20401.44335938,  115423.2109375 ,   39209.30859375]])
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mask_mismatch_dims( self ):

    actual = self.worldlines.data_masker.get_mask( np.array([ True, False, True, False ]) )
    expected = np.array([
      [ 1, 1, 1, ],
      [ 0, 0, 0, ],
      [ 1, 1, 1, ],
      [ 0, 0, 0, ],
    ])
    npt.assert_allclose( expected, actual )
