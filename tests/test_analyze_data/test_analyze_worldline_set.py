#!/usr/bin/env python
'''Testing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import numpy.testing as npt
import os
import h5py

import mock
import unittest

import pathfinder.analyze_data.worldline_set as analyze_worldline_set

########################################################################
# Commonly useful input variables

defaults = {
    'data_dir': './tests/data/tracking_output_for_analysis',
    'tag': 'analyze',
}

########################################################################


class TestWorldlineSetStartUp( unittest.TestCase ):

    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.__init__' )
    def test_init( self, mock_constructor ):

        variations = {
            'a': { 'data_dir': 'data_dir_a' },
            'b': { 'data_dir': 'data_dir_b' },
            'c': { 'tag': 'tag_c' },
            'd': {},
        }

        mock_constructor.side_effect = [ None, ] * len( variations )

        calls = [
            mock.call( data_dir='data_dir_a', tag=defaults['tag'], label='a' ),
            mock.call( data_dir='data_dir_b', tag=defaults['tag'], label='b' ),
            mock.call( data_dir=defaults['data_dir'], tag='tag_c', label='c' ),
            mock.call( data_dir=defaults['data_dir'], tag=defaults['tag'], label='d' ),
        ]

        worldline_set = analyze_worldline_set.WorldlineSet( defaults, variations )

        mock_constructor.assert_has_calls( calls, any_order=True )

        # Check that it behaves like a dict
        assert len( worldline_set ) == len( variations )
        for key in worldline_set.keys():
            assert key in variations
        for item in worldline_set:
            assert item in variations

########################################################################
########################################################################


class TestWorldlineSet( unittest.TestCase ):

    def setUp( self ):

        variations = {
            'a': { 'data_dir': 'data_dir_a' },
            'b': { 'data_dir': 'data_dir_b' },
            'c': { 'tag': 'tag_c' },
            'd': {},
        }

        self.worldline_set = analyze_worldline_set.WorldlineSet( defaults, variations )

    ########################################################################

    def test_getattr( self ):

        actual = self.worldline_set.data_object.data_dir
        expected = { 'a': 'data_dir_a', 'b': 'data_dir_b', 'c': defaults['data_dir'], 'd': defaults['data_dir'] }

        self.assertEqual( expected, actual )

    ########################################################################

    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.foo.bar', create=True, new=1 )
    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.foo', create=True )
    def test_getattr_nested( self, mock_foo ):

        actual = self.worldline_set.data_object.foo.bar
        expected = { 'a': 1, 'b': 1, 'c': 1, 'd': 1 }

        self.assertEqual( actual, expected )

    #########################################################################

    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.foo', create=True )
    def test_getmethod( self, mock_foo ):

        def side_effects( x ):
            return x

        mock_foo.side_effect = side_effects

        actual = self.worldline_set.data_object.foo( 1 )
        expected = { 'a': 1, 'b': 1, 'c': 1, 'd': 1 }

        self.assertEqual( actual, expected )

    #########################################################################

    @mock.patch( 'pathfinder.analyze_data.classifications.Classifications.__init__' )
    @mock.patch( 'pathfinder.analyze_data.classifications.Classifications.foo', create=True )
    def test_getmethod_nested( self, mock_foo, mock_constructor ):

        def side_effects( x, **kwargs ):
            return x

        mock_foo.side_effect = side_effects

        mock_constructor.side_effect = [ None, ] * len( self.worldline_set )

        actual = self.worldline_set.data_object.classifications.foo( 1, **{'t': 0} )
        expected = { 'a': 1, 'b': 1, 'c': 1, 'd': 1 }

        self.assertEqual( actual, expected )

    ########################################################################

    def test_from_tag_expansion( self ):
        '''Test alternate construction method.'''

        w_set = analyze_worldline_set.WorldlineSet.from_tag_expansion(
            defaults,
            'analyze_*',
        )

        # Make sure the tags are good
        assert w_set['analyze_v1'].tag == 'analyze_v1'
        assert w_set['analyze_v2'].tag == 'analyze_v2'

        # Finally, just try opening ptracks
        w_set.ptracks

    ########################################################################

########################################################################
########################################################################

class TestStoreQuantity( unittest.TestCase ):

    ########################################################################

    def setUp( self ):

        # Setup data
        variations = {
            'analyze_v1': { 'tag': 'analyze_v1' },
            'analyze_v2': { 'tag': 'analyze_v2' },
        }
        self.w_set = analyze_worldline_set.WorldlineSet( defaults, variations )

        self.stored_data_file = os.path.join(
            defaults['data_dir'], 'stored_quantity.hdf5' )

        if os.path.isfile( self.stored_data_file ):
            os.remove( self.stored_data_file )

    ########################################################################

    def test_store_quantity( self ):

        self.w_set.store_quantity(
            self.stored_data_file,
            selection_routine = None,
            sl = (slice(None),1),
        )

        f = h5py.File( self.stored_data_file, 'r' )

        expected_tags = np.array( [ 'analyze_v1', 'analyze_v2', ] )
        expected_fresh_acc = np.array( [ 0.50052142, 0.25078213 ] )

        for i,tag in enumerate( f['tags'][...] ):
            assert f['tags'][i]  in expected_tags
        npt.assert_allclose( f['is_fresh_accretion'], expected_fresh_acc )

    ########################################################################

    def test_store_quantity_variable_args( self ):
        '''Test that this works when giving variations to the arguments called.
        '''

        self.w_set.store_quantity(
            self.stored_data_file,
            selection_routine = None,
            quantity_method = 'get_categories_selected_quantity',
            variations = {
                'analyze_v1' : { 'sl' : (slice(None),1), },
                'analyze_v2' : { 'sl' : (slice(None),2), },
            },
        )

        f = h5py.File( self.stored_data_file, 'r' )

        expected_tags = np.array( [ 'analyze_v1', 'analyze_v2', ] )
        expected_fresh_acc = np.array( [ 21203.41601562, 7096.78808594 ] )

        for i,tag in enumerate( f['tags'][...] ):
            assert f['tags'][i]  in expected_tags
        npt.assert_allclose( f['is_fresh_accretion'], expected_fresh_acc )


