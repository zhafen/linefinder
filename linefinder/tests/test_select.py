#!/usr/bin/env python
'''Testing for select.py

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
import unittest

import linefinder.config as config
import linefinder.select as select

import galaxy_dive.utils.utilities as utilities

########################################################################

# For IDSelector
default_kwargs = {
    'snum_start': 500,
    'snum_end': 600,
    'snum_step': 100,
    'p_types': [0, 4],
    'out_dir': './tests/data/tracking_output',
    'tag': 'test',

    'snapshot_kwargs': {
        'sdir': './tests/data/stars_included_test_data',
        'load_additional_ids': True,
        'ahf_index': 600,
        'halo_data_dir': './tests/data/ahf_test_data',
        'length_scale_used': 'Rvir',
    },
}

# For SnapshotIDSelector
default_snap_kwargs = {
    'sdir': './tests/data/stars_included_test_data',
    'snum': 500,
    'ptype': 0,
    'load_additional_ids': False,
    'ahf_index': 600,
    'halo_data_dir': './tests/data/ahf_test_data',
    'length_scale_used': 'Rvir',
}

newids_snap_kwargs = copy.deepcopy( default_snap_kwargs )
newids_snap_kwargs['load_additional_ids'] = True

default_data_filters = {
    'radial_cut': { 'data_key': 'Rf', 'data_min': 0., 'data_max': 1., },
    'temp_cut': { 'data_key': 'T', 'data_min': 1e4, 'data_max': 1e6, },
}

id_sampler_kwargs = {
    'out_dir': './tests/data/tracking_output_for_analysis',
    'tag': 'test',
    'n_samples': 3,
}

########################################################################
########################################################################


class TestSnapshotIDSelector( unittest.TestCase ):

    def setUp( self ):

        self.snapshot_id_selector = select.SnapshotIDSelector( **default_snap_kwargs )

        # Setup some test data with a range of values useful to us.
        self.snapshot_id_selector.data['R'] = np.array( [ 0.5, 1.2, 0.75, 0.1, 0.3, 1.5 ] ) * self.snapshot_id_selector.length_scale
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
            'Rf': np.array( [ 0, 1, 0, 0, 0, 1 ] ).astype( bool ),
            'T': np.array( [ 1, 0, 1, 0, 0, 0 ] ).astype( bool ),
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

        self.snapshot_id_selector = select.SnapshotIDSelector( **newids_snap_kwargs )

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

        # Mock the code version so we don't repeatedly change test data
        patcher = mock.patch( 'galaxy_dive.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        self.side_effects = [
            set( [ (10952235, 0), (36091289, 893109954) ] ),
            set( [ (10952235, 0), (123456, 35) ] ),
            set( [ (1573, 0), (12, 35), (15, 4), (0, 0) ] ),
            set(),
        ]

        self.id_selector = select.IDSelector( **default_kwargs )

        self.selected_ids = set( [ (10952235, 0), (36091289, 893109954), (123456, 35), (1573, 0), (12, 35), (15, 4), (0, 0) ])
        self.selected_ids_formatted = np.array([ 0, 10952235, 15, 1573, 36091289, 123456, 12 ])
        self.selected_child_ids_formatted = np.array([ 0, 0, 4, 0, 893109954, 35, 35 ])

    ########################################################################

    @mock.patch( 'linefinder.select.SnapshotIDSelector.__init__' )
    @mock.patch( 'linefinder.select.SnapshotIDSelector.select_ids_snapshot' )
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
        calls = [ mock.call( **call_kwarg ) for call_kwarg in call_kwargs ]

        # Actually run the thing
        actual = self.id_selector.get_selected_ids( default_data_filters )
        expected = self.selected_ids
        assert expected == actual

        mock_constructor.assert_has_calls( calls )

    ########################################################################

    @mock.patch( 'linefinder.select.SnapshotIDSelector.__init__' )
    @mock.patch( 'linefinder.select.IDSelector.get_selected_ids_snapshot' )
    def test_get_selected_ids_parallel( self, mock_get_selected_ids_snapshot, mock_constructor, ):

        self.id_selector.n_processors = 2

        # Mock setup
        mock_constructor.side_effect = [ None, ]*4
        def side_effects( args ):
            kwargs = args[1]
            if kwargs['snum'] == 500:
                if kwargs['ptype'] == 0:
                    return set( [ (10952235, 0), (36091289, 893109954) ] )
                elif kwargs['ptype'] == 4:
                    return set( [ (10952235, 0), (123456, 35) ] )
            if kwargs['snum'] == 600:
                if kwargs['ptype'] == 0:
                    return set( [ (1573, 0), (12, 35), (15, 4), (0, 0) ] )
                elif kwargs['ptype'] == 4:
                    return set()
        mock_get_selected_ids_snapshot.side_effect = side_effects


        call_kwargs = [ copy.deepcopy( newids_snap_kwargs ) for i in range(4) ]
        call_kwargs[0]['snum'] = 500
        call_kwargs[1]['snum'] = 500
        call_kwargs[2]['snum'] = 600
        call_kwargs[3]['snum'] = 600
        call_kwargs[0]['ptype'] = 0
        call_kwargs[1]['ptype'] = 4
        call_kwargs[2]['ptype'] = 0
        call_kwargs[3]['ptype'] = 4
        calls = [ mock.call( **call_kwarg ) for call_kwarg in call_kwargs ]

        # Actually run the thing
        actual = self.id_selector.get_selected_ids_parallel( default_data_filters )
        expected = self.selected_ids
        assert expected == actual

    ########################################################################

    def test_format_selected_ids( self ):

        actual = self.id_selector.format_selected_ids( self.selected_ids )
        expected = [ self.selected_ids_formatted, self.selected_child_ids_formatted ]

        for i, id_ in enumerate( expected[0] ):
            assert id_ in actual[0]
            match_actual = np.where(actual[0]==id_)
            npt.assert_allclose( actual[1][match_actual], expected[1][i] )

    ########################################################################

    def test_save_selected_ids( self ):

        # Make sure there's nothing in our way, bwahahah
        ids_filepath = './tests/data/tracking_output/ids_full_test.hdf5'
        if os.path.isfile( ids_filepath ):
            os.system( 'rm {}'.format( ids_filepath ) )

        # The function itself
        ids = ( self.selected_ids_formatted, self.selected_child_ids_formatted )
        data_filters = {
            'radial_cut': {
                'data_key': 'Rf',
                'data_min': 0.,
                'data_max': 1.,
            },
            'velocity_cut': {
                'data_key': 'Vf',
                'data_min': 0.,
                'data_max': 0.5,
            },
        }
        self.id_selector.save_selected_ids( ids, data_filters )

        filepath = os.path.join( default_kwargs['out_dir'], 'ids_full_test.hdf5' )
        g = h5py.File( filepath, 'r' )

        expected = self.selected_ids_formatted
        actual = g['target_ids'][...]
        npt.assert_allclose( expected, actual )

        expected = self.selected_child_ids_formatted
        actual = g['target_child_ids'][...]
        npt.assert_allclose( expected, actual )

        for key in default_kwargs.keys():
            if key == 'snapshot_kwargs':
                snapshot_kwargs = default_kwargs[key]
                for key2 in snapshot_kwargs.keys():
                    assert snapshot_kwargs[key2] == g['parameters/snapshot_parameters'].attrs[key2]
            elif key == 'p_types':
                continue
            else:
                assert default_kwargs[key] == g['parameters'].attrs[key]

        for key, data_filter in data_filters.items():
            for inner_key, data_filter_inner in data_filter.items():
                assert data_filter_inner == g['parameters/data_filters'][key].attrs[inner_key]

        assert g.attrs['linefinder_version'] is not None
        assert g.attrs['galaxy_dive_version'] is not None

    ########################################################################

    @mock.patch( 'linefinder.select.SnapshotIDSelector.__init__' )
    @mock.patch( 'linefinder.select.SnapshotIDSelector.select_ids_snapshot' )
    def test_select_ids( self, mock_select_ids_snapshot, mock_constructor, ):

        # Make sure there's nothing in our way, bwahahah
        ids_filepath = './tests/data/tracking_output/ids_full_test.hdf5'
        if os.path.isfile( ids_filepath ):
            os.system( 'rm {}'.format( ids_filepath ) )

        # Mock setup
        mock_constructor.side_effect = [ None, ] * 4
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
        calls = [ mock.call( **call_kwarg ) for call_kwarg in call_kwargs ]

        # Actually run it
        self.id_selector.select_ids( default_data_filters )

        mock_constructor.assert_has_calls( calls )

        # Do the same tests as saving the data at the end
        filepath = os.path.join( default_kwargs['out_dir'], 'ids_full_test.hdf5' )
        g = h5py.File( filepath, 'r' )

        expected = self.selected_ids
        actual_ids = g['target_ids'][...]
        actual_child_ids = g['target_child_ids'][...]
        actual = set( zip( actual_ids, actual_child_ids ) )
        assert expected == actual

        for key in default_kwargs.keys():
            if key == 'snapshot_kwargs':
                snapshot_kwargs = default_kwargs[key]
                for key2 in snapshot_kwargs.keys():
                    assert snapshot_kwargs[key2] == g['parameters/snapshot_parameters'].attrs[key2]
            elif key == 'p_types':
                continue
            else:
                assert default_kwargs[key] == g['parameters'].attrs[key]

        assert g.attrs['linefinder_version'] is not None
        assert g.attrs['galaxy_dive_version'] is not None

########################################################################

class TestIDSelectorNoChildIDs( unittest.TestCase ):

    def setUp( self ):

        # Mock the code version so we don't repeatedly change test data
        patcher = mock.patch( 'galaxy_dive.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        self.side_effects = [
            set( [ 10952235, 36091289, ] ),
            set( [ 10952235, 123456,  ] ),
            set( [ 1573, 12, 15, 0, ] ),
            set(),
        ]

        self.id_selector = select.IDSelector( **default_kwargs )

        self.selected_ids = set( [ 10952235, 36091289, 123456, 1573, 12, 15, 0, ])
        self.selected_ids_formatted = np.array([ 0, 10952235, 15, 1573, 36091289, 123456, 12 ])

    ########################################################################

    @mock.patch( 'linefinder.select.SnapshotIDSelector.__init__' )
    @mock.patch( 'linefinder.select.SnapshotIDSelector.select_ids_snapshot' )
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
        calls = [ mock.call( **call_kwarg ) for call_kwarg in call_kwargs ]

        # Actually run the thing
        actual = self.id_selector.get_selected_ids( default_data_filters )
        expected = self.selected_ids
        assert expected == actual

        mock_constructor.assert_has_calls( calls )

    ########################################################################

    @mock.patch( 'linefinder.select.SnapshotIDSelector.__init__' )
    @mock.patch( 'linefinder.select.IDSelector.get_selected_ids_snapshot' )
    def test_get_selected_ids_parallel( self, mock_get_selected_ids_snapshot, mock_constructor, ):

        self.id_selector.n_processors = 2

        # Mock setup
        mock_constructor.side_effect = [ None, ]*4
        def side_effects( args ):
            kwargs = args[1]
            if kwargs['snum'] == 500:
                if kwargs['ptype'] == 0:
                    return set( [ 10952235, 36091289, ] )
                elif kwargs['ptype'] == 4:
                    return set( [ 10952235, 123456,  ] )
            if kwargs['snum'] == 600:
                if kwargs['ptype'] == 0:
                    return set( [ 1573, 12, 15, 0,  ] )
                elif kwargs['ptype'] == 4:
                    return set()
        mock_get_selected_ids_snapshot.side_effect = side_effects


        call_kwargs = [ copy.deepcopy( newids_snap_kwargs ) for i in range(4) ]
        call_kwargs[0]['snum'] = 500
        call_kwargs[1]['snum'] = 500
        call_kwargs[2]['snum'] = 600
        call_kwargs[3]['snum'] = 600
        call_kwargs[0]['ptype'] = 0
        call_kwargs[1]['ptype'] = 4
        call_kwargs[2]['ptype'] = 0
        call_kwargs[3]['ptype'] = 4
        calls = [ mock.call( **call_kwarg ) for call_kwarg in call_kwargs ]

        # Actually run the thing
        actual = self.id_selector.get_selected_ids_parallel( default_data_filters )
        expected = self.selected_ids
        assert expected == actual

    ########################################################################

    def test_format_selected_ids( self ):

        actual = self.id_selector.format_selected_ids( copy.copy( self.selected_ids ) )
        expected = self.selected_ids_formatted

        # Sort the arrays, because the order doesn't really matter anyways for the single ID case
        npt.assert_allclose( np.sort( expected ), np.sort( actual ) )

    ########################################################################

    def test_save_selected_ids( self ):

        # Make sure there's nothing in our way, bwahahah
        ids_filepath = './tests/data/tracking_output/ids_full_test.hdf5'
        if os.path.isfile( ids_filepath ):
            os.system( 'rm {}'.format( ids_filepath ) )

        # The function itself
        self.id_selector.save_selected_ids( self.selected_ids_formatted, default_data_filters )

        filepath = os.path.join( default_kwargs['out_dir'], 'ids_full_test.hdf5' )
        g = h5py.File( filepath, 'r' )

        expected = self.selected_ids_formatted
        actual = g['target_ids'][...]
        npt.assert_allclose( expected, actual )

        for key in default_kwargs.keys():
            if key == 'snapshot_kwargs':
                snapshot_kwargs = default_kwargs[key]
                for key2 in snapshot_kwargs.keys():
                    assert snapshot_kwargs[key2] == g['parameters/snapshot_parameters'].attrs[key2]
            elif key == 'p_types':
                continue
            else:
                assert default_kwargs[key] == g['parameters'].attrs[key]

        assert g.attrs['linefinder_version'] is not None
        assert g.attrs['galaxy_dive_version'] is not None

########################################################################
########################################################################

class TestIDSamplerNoSetUp( unittest.TestCase ):

    def setUp( self ):

        self.filepath = os.path.join( id_sampler_kwargs['out_dir'], 'ids_test.hdf5' )
        self.id_sampler = select.IDSampler( **id_sampler_kwargs )

        if os.path.isfile( self.filepath ):
            os.remove( self.filepath )

    ########################################################################

    def test_copy_and_open_full_ids( self ):

        self.id_sampler.copy_and_open_full_ids()

        expected = np.array([       0, 10952235,       15,     1573, 36091289,   123456,       12])
        actual = self.id_sampler.f['target_ids'][...]
        npt.assert_allclose( expected, actual )

        expected = self.filepath
        actual = self.id_sampler.f.filename
        assert os.path.samefile( expected, actual )

    ########################################################################

    def test_sample_ids( self ):

        np.random.seed( 1234 )

        self.id_sampler.sample_ids()

        g = h5py.File( self.filepath, 'r' )

        expected = np.array([      15, 10952235,       12])
        actual = g['target_ids']
        npt.assert_allclose( expected, actual )

        assert g['parameters'].attrs['n_samples'] == 3
        assert g['parameters'].attrs['ignore_child_particles'] == False
        assert g['parameters'].attrs['sampled_from_full_id_list']

########################################################################
########################################################################

class TestIDSampler( unittest.TestCase ):

    def setUp( self ):

        self.filepath = os.path.join( id_sampler_kwargs['out_dir'], 'ids_test.hdf5' )
        self.id_sampler = select.IDSampler( **id_sampler_kwargs )

        np.random.seed( 1234 )

        if os.path.isfile( self.filepath ):
            os.remove( self.filepath )

        self.id_sampler.copy_and_open_full_ids()

    ########################################################################

    def test_choose_particles_to_sample( self ):

        self.id_sampler.choose_particles_to_sample()

        actual = self.id_sampler.ids_to_sample
        expected = np.array([       0, 10952235,       15,     1573, 36091289,   123456,       12])
        actual.sort() ; expected.sort() # Sets can put things out of order.
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_choose_particles_to_sample_child_ids_included( self ):

        # Create some child ids
        self.id_sampler.f['target_child_ids'] = np.array([ 4, 0, 0, 2, 0, 0, 8 ])

        self.id_sampler.choose_particles_to_sample()

        actual = self.id_sampler.child_ids_to_sample
        expected = np.array([ 4, 0, 0, 2, 0, 0, 8 ])
        actual.sort() ; expected.sort() # Sets can put things out of order.
        npt.assert_allclose( expected, actual )

        actual = self.id_sampler.ids_to_sample
        expected = np.array([       0, 10952235,       15,     1573, 36091289,   123456,       12])
        actual.sort() ; expected.sort() # Sets can put things out of order.
        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'linefinder.select.IDSampler.identify_duplicate_ids' )
    def test_choose_particles_to_sample_ignore_duplicates( self, mock_identify_duplicate_ids ):

        mock_identify_duplicate_ids.side_effect = [ set([ 0, 15, 12 ]), ]

        self.id_sampler.ignore_duplicates = True
        self.id_sampler.ignore_child_particles = False

        self.id_sampler.choose_particles_to_sample()

        mock_identify_duplicate_ids.assert_called_once()

        actual = self.id_sampler.ids_to_sample
        expected = np.array([      10952235,     1573, 36091289,   123456, ])
        actual.sort() ; expected.sort() # Sets can put things out of order.
        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'linefinder.select.IDSampler.identify_child_particles' )
    def test_choose_particles_to_sample_ignore_child_particles( self, mock_identify_child_particles ):

        # Create some child ids
        self.id_sampler.f['target_child_ids'] = np.array([ 4, 0, 0, 2, 0, 0, 8 ])

        mock_identify_child_particles.side_effect = [ set([ ( 0, 4), ( 15, 0 ), ( 12, 8 ), ]), ]

        self.id_sampler.ignore_duplicates = False
        self.id_sampler.ignore_child_particles = True

        self.id_sampler.choose_particles_to_sample()

        mock_identify_child_particles.assert_called_once()

        actual = self.id_sampler.ids_to_sample
        expected = np.array([      10952235,     1573, 36091289,   123456, ])
        actual.sort() ; expected.sort() # Sets can put things out of order.
        npt.assert_allclose( expected, actual )

        actual = self.id_sampler.child_ids_to_sample
        expected = np.array([ 0, 2, 0, 0, ])
        actual.sort() ; expected.sort() # Sets can put things out of order.
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_choose_sample_inds( self ):

        self.id_sampler.ids_to_sample = np.array([       0, 10952235,       15,     1573, 36091289,   123456,       12])

        self.id_sampler.choose_sample_inds()

        expected = np.array([ 2, 1, 6, ])
        actual = self.id_sampler.sample_inds
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_identify_child_particles( self ):

        # Create some child ids
        self.id_sampler.f['target_child_ids'] = np.array([ 4, 0, 0, 2, 0, 0, 8 ])

        actual = self.id_sampler.identify_child_particles()

        expected = set( [ ( 0, 4, ), ( 1573, 2 ), ( 12, 8, ) ] )

        self.assertEqual( expected, actual )

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.simulation_data.SnapshotData.get_data' )
    @mock.patch( 'galaxy_dive.analyze_data.particle_data.ParticleData.find_duplicate_ids' )
    @mock.patch( 'galaxy_dive.analyze_data.particle_data.ParticleData.__init__' )
    def test_identify_duplicate_ids_load_correct_data( self, mock_p_data, mock_find_duplicate_ids, mock_get_data ):

        mock_p_data.side_effect = [ None, None, ]

        actual = self.id_sampler.identify_duplicate_ids()

        expected_kwargs = dict( default_snap_kwargs )
        del expected_kwargs['halo_data_dir']
        expected_kwargs['load_additional_ids'] = True
        expected_kwargs['snum'] = 600
        expected_kwargs['analysis_dir'] = default_snap_kwargs['halo_data_dir']
        del expected_kwargs['length_scale_used']

        expected_kwargs0 = dict( expected_kwargs )
        expected_kwargs0['ptype'] = 0
        expected_kwargs4 = dict( expected_kwargs )
        expected_kwargs4['ptype'] = 4

        calls = [ mock.call( **expected_kwargs0 ), mock.call( **expected_kwargs4 ) ]

        mock_p_data.assert_has_calls( calls )

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.particle_data.ParticleData.find_duplicate_ids' )
    def test_identify_duplicate_ids( self, mock_find_duplicate_ids ):

        mock_find_duplicate_ids.side_effect = [ np.array([ 36091289, 3211791 ]), np.array([ 24565150, ]) ]

        actual = self.id_sampler.identify_duplicate_ids()

        expected = set( [ 24565150, 36091289, 3211791 ] )

        self.assertEqual( expected, actual )

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.simulation_data.SnapshotData.get_data' )
    @mock.patch( 'galaxy_dive.analyze_data.particle_data.ParticleData.find_duplicate_ids' )
    def test_identify_duplicate_ids_star_gas( self, mock_find_duplicate_ids, mock_get_data ):
        '''Test we can identify duplicates when one of the particles is gas, and the other is star.
        '''

        mock_find_duplicate_ids.side_effect = [ np.array([ 36091289, 3211791 ]), np.array([ 24565150, ]) ]
        mock_get_data.side_effect = [ np.array([ 1, 2, 3, ]), np.array([ 2, 3, 4, 5 ] ) ]

        actual = self.id_sampler.identify_duplicate_ids()

        expected = set( [ 24565150, 36091289, 3211791, 2, 3, ] )

        self.assertEqual( expected, actual )

    ########################################################################

    def test_save_sampled_ids( self ):

        self.id_sampler.ids_to_sample = np.array([       0, 10952235,       15,     1573, 36091289,   123456,       12])
        self.id_sampler.sample_inds =  np.array([ 1, 2, 4, ])

        self.id_sampler.save_sampled_ids()

        g = h5py.File( self.filepath, 'r' )

        expected = np.array([10952235,       15, 36091289])
        actual = g['target_ids']
        npt.assert_allclose( expected, actual )

        assert g['parameters'].attrs['n_samples'] == 3
        assert g['parameters'].attrs['ignore_child_particles'] == False
        assert g['parameters'].attrs['sampled_from_full_id_list']

    ########################################################################

    def test_save_sampled_ids_child_ids( self ):

        self.id_sampler.ids_to_sample = np.array([       0, 10952235,       15,     1573, 36091289,   123456,       12])
        self.id_sampler.f['target_child_ids'] = np.array([ 4, 0, 0, 2, 0, 0, 8 ])
        self.id_sampler.child_ids_to_sample = np.array([ 4, 0, 0, 2, 0, 0, 8 ])
        self.id_sampler.sample_inds =  np.array([ 0, 1, 4, ])

        self.id_sampler.save_sampled_ids()

        g = h5py.File( self.filepath, 'r' )

        expected = np.array([       0, 10952235, 36091289])
        actual = g['target_ids']
        npt.assert_allclose( expected, actual )

        expected = np.array([ 4, 0, 0, ])
        actual = g['target_child_ids']
        npt.assert_allclose( expected, actual )

        assert g['parameters'].attrs['n_samples'] == 3
        assert g['parameters'].attrs['ignore_child_particles'] == False
        assert g['parameters'].attrs['sampled_from_full_id_list']

    ########################################################################


class TestIDSelectorJug( unittest.TestCase ):

    def setUp( self ):

        # We can't execute from the package module
        os.chdir( '..' )

        self.out_dir = './linefinder/tests/data/tracking_output'

        # Remove output
        self.filepaths = []
        for end in [ 'ids_full_test.hdf5', 'ids_full_test_jug.hdf5' ]:
            filepath = os.path.join( self.out_dir, end )

            self.filepaths.append( filepath )

            if os.path.isfile( filepath ):
                os.remove( filepath )

    ########################################################################

    def tearDown( self ):

        # Remove jugdata
        os.system( 'rm -r ./linefinder/tests/*jugdata' )

        # Switch back so we don't mess up other tests
        os.chdir( 'linefinder' )

    ########################################################################

    def test_select_ids_jug( self ):

        kwargs = copy.copy( default_kwargs )

        # Modifications to the output dirs to account for shifting up one level
        kwargs['out_dir'] = self.out_dir
        kwargs['snapshot_kwargs']['sdir'] = './linefinder/tests/data/stars_included_test_data'
        kwargs['snapshot_kwargs']['halo_data_dir'] = './linefinder/tests/data/ahf_test_data'

        data_filters = {
            'radial_cut': { 'data_key': 'Rf', 'data_min': 0., 'data_max': 1., },
        }

        id_selector = select.IDSelector( **kwargs )
        id_selector.select_ids( data_filters )

        # Run jug version
        os.system( "jug execute ./linefinder/tests/select_jugfile.py &"
        )
        os.system( "jug execute ./linefinder/tests/select_jugfile.py"
        )

        files = []
        for filepath in self.filepaths:
            files.append( h5py.File( filepath, 'r' ) )

        for key in [ 'target_ids', 'target_child_ids' ]:

            npt.assert_allclose( files[0][key][...], files[1][key][...] )

        for key in files[0]['parameters'].attrs.keys():

            # These shouldn't match
            if key == 'tag':
                continue

            try:
                self.assertEqual(
                    utilities.check_and_decode_bytes(
                        files[0]['parameters'].attrs[key],
                    ),
                    utilities.check_and_decode_bytes(
                        files[1]['parameters'].attrs[key],
                    ),
                )
            # In case it's an array
            except ValueError:
                npt.assert_allclose(
                    files[0]['parameters'].attrs[key],
                    files[1]['parameters'].attrs[key],
                )
