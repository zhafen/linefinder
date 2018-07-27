#!/usr/bin/env python
'''Testing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import mock
import numpy as np
import numpy.testing as npt
import os
import unittest

import pathfinder.analyze_data.worldlines as analyze_worldlines
import pathfinder.config as config
import pathfinder.utils.presentation_constants as p_constants

########################################################################
# Commonly useful input variables

tracking_dir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

kwargs = {
    'ahf_data_dir': './tests/data/ahf_test_data',
    'ahf_index': 600,
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
            'data_dir': './tests/data/tracking_output_for_analysis',
            'tag': 'analyze',

            'ptracks_tag': 'alt_tag1',
            'galids_tag': 'alt_tag2',
            'classifications_tag': 'alt_tag3',

            'ahf_data_dir': './tests/data/ahf_test_data',
            'ahf_index': 600,
        }

        self.worldlines = analyze_worldlines.Worldlines( **self.kwargs )

    ########################################################################

    @mock.patch( 'pathfinder.analyze_data.classifications.Classifications.__init__' )
    @mock.patch( 'pathfinder.analyze_data.galids.GalIDs.__init__' )
    @mock.patch( 'pathfinder.analyze_data.ptracks.PTracks.__init__' )
    def test_different_tags( self, mock_ptracks, mock_galids, mock_classifications ):

        mock_ptracks.side_effect = [ None, ]
        mock_galids.side_effect = [ None, ]
        mock_classifications.side_effect = [ None, ]

        self.worldlines.ptracks
        ptrack_kwargs = {
            'ahf_data_dir': './tests/data/ahf_test_data',
            'ahf_index': 600,
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

        mock_get_data.assert_called_once_with( 'Rx' )

    ########################################################################

    def test_get_stored_data_is_R( self ):
        '''There's a bug that shows up when the stored data is 'R'.'''

        self.worldlines.data['R'] = np.array([ 0., 1., 2., 3., ])

        actual = self.worldlines.get_data( 'R' )

        npt.assert_allclose( self.worldlines.data['R'], actual )

    ########################################################################

    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.calc_method', create=True )
    def test_handle_data_key_error( self, mock_calc_method ):

        self.worldlines.handle_data_key_error( 'method' )

        mock_calc_method.assert_called_once()

    ########################################################################

    def test_get_fraction_outside( self ):

        # Setup test data
        self.worldlines.data['A'] = np.array([
            [ 1., 2., 3., ],
            [ 3., 3., 1., ],
            [ 1., 1., 1., ],
            [ 1., 3., 1., ],
        ])

        actual = self.worldlines.get_fraction_outside( 'A', 0., 2.5, )
        expected = 4./12.

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_data_first_acc( self ):

        self.worldlines.data['PType'] = np.array([
            [ 4, 4, 0, ],
            [ 4, 0, 0, ],
            [ 4, 0, 0, ],
            [ 4, 0, 0, ],
        ])
        self.worldlines.data['ind_first_acc'] = np.array([ 0, 0, 1, config.INT_FILL_VALUE, ])

        actual = self.worldlines.get_data_first_acc( 'PType', ind_after_first_acc=True )
        expected = np.array( [ 4, 4, 0, config.INT_FILL_VALUE ] )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_data_first_acc_float( self ):

        self.worldlines.data['Den'] = np.array([
            [ 1., 2., 3., ],
            [ 1., 2., 3., ],
            [ 1., 2., 3., ],
            [ 1., 2., 3., ],
        ])
        self.worldlines.data['ind_first_acc'] = np.array([ 0, 0, 1, config.INT_FILL_VALUE, ])

        actual = self.worldlines.get_data_first_acc( 'Den', ind_after_first_acc=True )
        expected = np.array( [ 1., 1., 2., config.FLOAT_FILL_VALUE ] )
        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'galaxy_diver.analyze_data.ahf.HaloData.get_mt_data' )
    def test_get_data_at_ind_units_used( self, mock_get_mt_data ):
        '''Test we can get the data at a specified index, including scaling by some unit.
        '''

        # Setup test data
        self.worldlines.data['ind_test'] = np.array([ 0, 1, 2, 1, ])
        self.worldlines.data['test_data'] = np.array([
            [ 1., 2., 3., ],
            [ 2., 4., 6., ],
            [ 3., 6., 9., ],
            [ 4., 8., 12., ],
        ])
        mock_get_mt_data.side_effect = [
            np.array([ 1., 2., 3., ]),
        ]

        actual = self.worldlines.get_data_at_ind( 'test_data', 'ind_test', units='Rvir' )
        expected = np.array([ 1., 2., 3., 4., ])*0.702

        npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestWorldlineGetStellarMass( unittest.TestCase ):

    def setUp( self ):

        self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

        # Setup test data
        self.worldlines._n_particles = 6
        self.worldlines.ptracks.data['M'] = np.array([
            [ 1., 1., 1., ],
            [ 1., 2., 1., ],
            [ 1., 3., 1., ],
            [ 1., 4., 1., ],
            [ 1., 5., 1., ],
            [ 1., 6., 1., ],
        ])
        self.worldlines.data['PType'] = np.array([
            [ 4, 4, 4, ],
            [ 4, 4, 0, ],
            [ 4, 4, 0, ],
            [ 4, 4, 0, ],
            [ 4, 0, 0, ],
            [ 4, 4, 0, ],
        ])
        self.worldlines.events.data['is_in_main_gal'] = np.array([
            [ 1, 1, 1, ],
            [ 1, 1, 0, ],
            [ 1, 1, 0, ],
            [ 1, 1, 0, ],
            [ 1, 1, 0, ],
            [ 1, 0, 0, ],
        ])
        self.worldlines.classifications.data['is_merger'] = np.array([
            1, 0, 0, 0, 0, 0, ]).astype( bool )
        self.worldlines.classifications.data['is_mass_transfer'] = np.array([
            0, 1, 0, 0, 0, 0,  ]).astype( bool )
        self.worldlines.data['is_fresh_accretion'] = np.array([
            0, 0, 1, 0, 1, 1, ]).astype( bool )
        self.worldlines.data['is_NEP_wind_recycling'] = np.array([
            0, 0, 0, 1, 0, 0, ]).astype( bool )

    ########################################################################

    def test_get_selected_quantity( self ):

        actual = self.worldlines.get_categories_selected_quantity_fraction(
            sl = (slice(None),1),
            classification_list = p_constants.CLASSIFICATIONS_B,
        )
        expected = {
            'is_merger': 0.1,
            'is_mass_transfer': 0.2,
            'is_fresh_accretion': 0.3,
            'is_NEP_wind_recycling': 0.4,
        }

        for key, item in expected.items():
            npt.assert_allclose( item, actual[key] )

    ########################################################################

    def test_get_selected_quantity_redshift( self ):

        actual = self.worldlines.get_categories_selected_quantity_fraction(
            classification_list = p_constants.CLASSIFICATIONS_B,
        )
        expected = {
            'is_merger': np.array([ 1./6., 0.1, 1., ]),
            'is_mass_transfer': np.array([ 1./6., 0.2, 0., ]),
            'is_fresh_accretion': np.array([ 0.5, 0.3, 0., ]),
            'is_NEP_wind_recycling': np.array([ 1./6., 0.4, 0., ]),
        }

        for key, item in expected.items():
            npt.assert_allclose( item, actual[key] )

    ########################################################################

    def test_get_selected_quantity_n_particles( self ):

        actual = self.worldlines.get_categories_selected_quantity(
            quantity = 'n_particles',
            classification_list = p_constants.CLASSIFICATIONS_B,
        )
        expected = {
            'is_merger': np.array([ 1, 1, 1, ]),
            'is_mass_transfer': np.array([ 1, 1, 0, ]),
            'is_fresh_accretion': np.array([ 3, 1, 0, ]),
            'is_NEP_wind_recycling': np.array([ 1, 1, 0, ]),
        }

        for key, item in expected.items():
            npt.assert_allclose( item, actual[key], err_msg="Key {} failed".format( key ) )

########################################################################
########################################################################

class TestWorldlineGetProcessedData( unittest.TestCase ):

    def setUp( self ):

        self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

    ########################################################################

    def test_basic( self ):

        # Setup test data
        self.worldlines.ptracks.data['A'] = np.array( [ 10., 100., 1000. ] )

        actual = self.worldlines.get_processed_data( 'logA' )
        expected = np.array( [ 1., 2., 3. ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_tiled_horizontal( self ):
        '''Test that we can add the tiled key and get out a tiled result.
        In this case, we're testing if we properly tile when we have one value per particle.
        '''

        # Setup test data
        self.worldlines.ptracks.data['A'] = np.array( [ 10., 100., 1000., 10000. ] )

        actual = self.worldlines.get_processed_data( 'logA_tiled' )
        expected = np.array([
            [ 1., 2., 3., 4., ],
            [ 1., 2., 3., 4., ],
            [ 1., 2., 3., 4., ],
        ]).transpose()

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_tiled_vertical( self ):
        '''Test that we can add the tiled key and get out a tiled result.
        In this case, we're testing if we properly tile when we have one value per particle.
        '''

        # Setup test data
        self.worldlines.ptracks.data['A'] = np.array( [ 10., 100., 1000., ] )

        actual = self.worldlines.get_processed_data( 'logA_tiled' )
        expected = np.array([
            [ 1., 2., 3., ],
            [ 1., 2., 3., ],
            [ 1., 2., 3., ],
            [ 1., 2., 3., ],
        ])

        npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestWorldlineCalcData( unittest.TestCase ):

    def setUp( self ):

        self.worldlines = analyze_worldlines.Worldlines( tracking_dir, tag, **kwargs )

    ########################################################################

    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.get_processed_data' )
    def test_is_in_CGM( self, mock_get_processed_data ):

        mock_get_processed_data.side_effect = [
            np.array([ 
                [ 0.05, 0.15, 1.1 ],
                [ 0.05, 0.15, 1.1 ],
                [ 0.5, 0.05, 0.5 ],
                [ 0.5, 0.05, 0.5 ],
            ]),
            np.array([ 
                [ 2, 10, 10 ],
                [ 10, 3, 10 ],
                [ 5, 5, 10 ],
                [ 2, 10, 10 ],
            ])
        ]

        expected = np.array([
            [ 0, 1, 0, ],
            [ 0, 0, 0, ],
            [ 0, 0, 1, ],
            [ 0, 0, 1, ],
        ]).astype( bool )

        self.worldlines.calc_is_in_CGM()

        actual = self.worldlines.data['is_in_CGM']

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_is_IP( self ):

        # Setup test data
        self.worldlines.data['is_in_main_gal'] = np.array([
            [ 0, 0, 0, ],
            [ 1, 0, 0, ],
            [ 1, 0, 1, ],
            [ 0, 1, 0, ],
        ]).astype( bool )

        expected = np.array([
            [ 0, 0, 0, ],
            [ 1, 0, 0, ],
            [ 1, 1, 1, ],
            [ 1, 1, 0, ],
        ]).astype( bool )

        self.worldlines.calc_is_IP()

        actual = self.worldlines.data['is_IP']

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_fresh_accretion( self ):

        # Setup test data
        self.worldlines.data['is_pristine'] = np.array( [ 1, 1, 1, 0, ] ).astype( bool )
        self.worldlines.data['is_wind'] = np.array([
            [ 1, 1, 0, ],
            [ 1, 0, 0, ],
            [ 0, 0, 0, ],
            [ 1, 1, 0, ],
        ]).astype( bool )

        self.worldlines.calc_is_fresh_accretion()

        actual = self.worldlines.data['is_fresh_accretion']
        expected = np.array([
            [ 0, 0, 1, ],
            [ 0, 1, 1, ],
            [ 1, 1, 1, ],
            [ 0, 0, 0, ],
        ]).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_NEP_wind_recycling( self ):

        # Setup test data
        self.worldlines.data['is_pristine'] = np.array( [ 1, 1, 1, 0, ] ).astype( bool )
        self.worldlines.data['is_wind'] = np.array([
            [ 1, 1, 0, ],
            [ 1, 0, 0, ],
            [ 0, 0, 0, ],
            [ 1, 1, 0, ],
        ]).astype( bool )

        self.worldlines.calc_is_NEP_wind_recycling()

        actual = self.worldlines.data['is_NEP_wind_recycling']
        expected = np.array([
            [ 1, 1, 0, ],
            [ 1, 0, 0, ],
            [ 0, 0, 0, ],
            [ 0, 0, 0, ],
        ]).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.get_data_first_acc' )
    def test_calc_is_merger_star( self, mock_get_data_first_acc ):

        # Setup test data
        mock_get_data_first_acc.side_effect = [ np.array( [ 4, 0, 4, 0, ] ), ]
        self.worldlines.data['is_merger'] = np.array( [ 1, 1, 0, 0, ] ).astype( bool )

        self.worldlines.calc_is_merger_star()

        actual = self.worldlines.data['is_merger_star']
        expected = np.array( [ True, False, False, False, ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'pathfinder.analyze_data.worldlines.Worldlines.get_data_first_acc' )
    def test_calc_is_merger_star( self, mock_get_data_first_acc ):

        # Setup test data
        mock_get_data_first_acc.side_effect = [ np.array( [ 4, 0, 4, 0, ] ), ]
        self.worldlines.data['is_merger'] = np.array( [ 1, 1, 0, 0, ] ).astype( bool )

        self.worldlines.calc_is_merger_gas()

        actual = self.worldlines.data['is_merger_gas']
        expected = np.array( [ False, True, False, False, ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_NEP_NYA( self ):
        '''Test that we can find material classified as NEP that is not yet accreted (NYA) onto the main galaxy.
        '''

        # Setup test data
        self.worldlines._n_snaps = 4
        self.worldlines.data['is_pristine'] = np.array([
            True,
            True,
            True,
            False,
        ])
        self.worldlines.data['ind_first_acc'] = np.array([
            0,
            2,
            1,
            1,
        ])

        self.worldlines.calc_is_NEP_NYA()

        actual = self.worldlines.data['is_NEP_NYA']
        expected = np.array([
            [ 0, 1, 1, 1, ],
            [ 0, 0, 0, 1, ],
            [ 0, 0, 1, 1, ],
            [ 0, 0, 0, 0, ],
        ]).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_after_enrichment( self ):
        '''Test that we can find when a particle is enriched.
        '''

        # Setup test data
        self.worldlines.data['Z'] = 10.**np.array([
            [ -2., -4., -4., ],
            [ -1., -2., -4., ],
            [ -4., -4., -4., ],
            [ 0., -1., -2., ],
        ])

        expected = np.array([
            [ 1, 0, 0, ],
            [ 1, 1, 0, ],
            [ 0, 0, 0, ],
            [ 1, 1, 1, ],
        ]).astype( bool )

        # Actual calculation
        self.worldlines.calc_is_after_enrichment()

        actual = self.worldlines.data['is_after_enrichment']

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_before_enrichment( self ):
        '''Test that we can find when a particle is enriched.
        '''

        # Setup test data
        self.worldlines.data['Z'] = 10.**np.array([
            [ -2., -4., -4., ],
            [ -1., -2., -4., ],
            [ -4., -4., -4., ],
            [ 0., -1., -2., ],
        ])

        expected = np.array([
            [ 0, 1, 0, ],
            [ 0, 1, 1, ],
            [ 0, 0, 0, ],
            [ 0, 1, 1, ],
        ]).astype( bool )

        # Actual calculation
        self.worldlines.calc_is_before_enrichment()

        actual = self.worldlines.data['is_before_enrichment']

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_enriched( self ):
        '''Test that we can find when a particle is enriched.
        '''

        # Setup test data
        self.worldlines.data['Z'] = 10.**np.array([
            [ -2., -4., -4., ],
            [ -1., -2., -4., ],
            [ -4., -4., -4., ],
            [ 0., -1., -2., ],
        ])

        expected = np.array([
            [ 1, 1, 0, ],
            [ 1, 1, 1, ],
            [ 0, 0, 0, ],
            [ 1, 1, 1, ],
        ]).astype( bool )

        # Actual calculation
        self.worldlines.calc_is_enriched()

        actual = self.worldlines.data['is_enriched']

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_enriched_in_mgal( self ):
        '''Test that we can find when a particle is enriched, and inside
        the radius of the main galaxy.
        '''

        # Setup test data
        self.worldlines.data['is_enriched'] = np.array([
            [ 1, 1, 0, ],
            [ 1, 1, 1, ],
            [ 0, 0, 0, ],
            [ 1, 1, 1, ],
        ]).astype( bool )
        self.worldlines.data['gal_id'] = np.array([
            [ -2, 15, -2, ],
            [ 5, -2, 1, ],
            [ -2, -2, -2, ],
            [ -2, -2, 5, ],
        ])
        self.worldlines.data['mt_gal_id'] = np.array([
            [ -2, -2, -2, ],
            [ 0, -2, 1, ],
            [ -2, -2, -2, ],
            [ -2, -2, 0, ],
        ])

        expected = np.array([
            [ 0, 0, 0, ],
            [ 1, 0, 0, ],
            [ 0, 0, 0, ],
            [ 0, 0, 1, ],
        ]).astype( bool )

        # Actual calculation
        self.worldlines.calc_is_enriched_in_mgal()

        actual = self.worldlines.data['is_enriched_in_mgal']

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_is_enriched_in_ogal( self ):
        '''Test that we can find when a particle is enriched, and inside
        the radius of another galaxy.
        '''

        # Setup test data
        self.worldlines.data['is_enriched'] = np.array([
            [ 1, 1, 0, ],
            [ 1, 1, 1, ],
            [ 0, 0, 0, ],
            [ 1, 1, 1, ],
        ]).astype( bool )
        self.worldlines.data['gal_id'] = np.array([
            [ -2, 15, -2, ],
            [ 5, -2, 1, ],
            [ -2, -2, -2, ],
            [ -2, -2, 5, ],
        ])
        self.worldlines.data['mt_gal_id'] = np.array([
            [ -2, -2, -2, ],
            [ 0, -2, 1, ],
            [ -2, -2, -2, ],
            [ -2, -2, 0, ],
        ])

        expected = np.array([
            [ 0, 1, 0, ],
            [ 0, 0, 1, ],
            [ 0, 0, 0, ],
            [ 0, 0, 0, ],
        ]).astype( bool )

        # Actual calculation
        self.worldlines.calc_is_enriched_in_ogal()

        actual = self.worldlines.data['is_enriched_in_ogal']

        npt.assert_allclose( expected, actual )

    ########################################################################

# This test is no longer relevant
#     def test_calc_ind_first_in_mgal( self ):
# 
#         # Setup test data
#         self.worldlines.data['ind_first_acc'] = np.array([
#             -2,
#             -2,
#             1,
#             1,
#         ])
#         self.worldlines.data['is_enriched_in_mgal'] = np.array([
#             [ 0, 0, 0, ],
#             [ 0, 1, 0, ],
#             [ 1, 0, 0, ],
#             [ 1, 1, 1, ],
#         ]).astype( bool )
# 
#         expected = np.array([
#             -2,
#             1,
#             1,
#             2,
#         ])
# 
#         # Actual calculation
#         self.worldlines.calc_ind_first_in_mgal()
# 
#         actual = self.worldlines.data['ind_first_in_mgal']
# 
#         npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_dt( self ):

        # Setup test data
        self.worldlines.classifications.data['redshift'] = np.array( [ 0., 0.06984670, 0.16946000, ] )

        self.worldlines.calc_dt()
        actual = self.worldlines.data['dt']
        expected = np.array( [ 0.927, 1.177, np.nan ] )

        npt.assert_allclose( expected, actual, rtol=1e-3 )

    ########################################################################

    def test_calc_t_EP( self ):
        '''Test basic functionality of calc_t_EP.'''

        # Setup test data
        self.worldlines._n_snaps = 4
        self.worldlines._n_particles = 3
        self.worldlines.data['is_in_other_gal'] = np.array([
            [ 0, 1, 1, 0, ],
            [ 0, 0, 1, 1, ],
            [ 0, 1, 0, 0, ],
        ]).astype( bool )
        self.worldlines.data['ind_first_acc'] = np.array([
            1,
            0,
            2,
        ])
        self.worldlines.data['dt'] = np.array([
            1.,
            2.,
            3.,
            4.,
        ])

        # Actual calculation
        self.worldlines.calc_t_EP()

        actual = self.worldlines.data['t_EP']
        expected = np.array([
            3000.,
            7000.,
            0.,
        ])

        npt.assert_allclose( expected, actual,)

    ########################################################################

    def test_calc_d_sat_scaled_min( self ):

        # Setup test data
        self.worldlines.data['d_sat_scaled'] = np.array([
            [ 2., 2., 1., ],
            [ 2., 0.5, 3., ],
            [ 1., 2., -2., ],
            [ 1., 2., 3., ],
        ])
        self.worldlines._redshift = np.array([ 1., 2., 3., ])
        self.worldlines.events.data['ind_first_acc'] = np.array([ 0, 0, 0, 1, ])

        self.worldlines.calc_d_sat_scaled_min()

        actual = self.worldlines.data['d_sat_scaled_min']
        expected = np.array([ 1., 0.5, 2., 3., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_ind_star( self ):

        # Setup test data
        self.worldlines.data['PType'] = np.array([
            [ 4, 0, 0, ],
            [ 0, 0, 0, ],
            [ 4, 4, 0, ],
            [ 4, 4, 4, ],
        ])

        self.worldlines.calc_ind_star()

        actual = self.worldlines.data['ind_star']
        expected = np.array([ 0, config.INT_FILL_VALUE, 1, -1, ])

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

        self.worldlines.events.data['ind_first_acc'] = np.array([ 2, config.INT_FILL_VALUE, 0, 2 ])

        actual = self.worldlines.data_masker.get_masked_data( 'T', mask_after_first_acc=True,  classification='is_preprocessed', sl=( slice(None), slice(0,2), ) )
        expected = np.array( [ 12212.33984375,   812602.1875, 4107.27490234, ] )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_masked_data_after_first_acc( self ):

        self.worldlines.events.data['ind_first_acc'] = np.array([ 2, config.INT_FILL_VALUE, 0, 2 ])

        actual = self.worldlines.data_masker.get_masked_data( 'T', mask_before_first_acc=True,  classification='is_preprocessed', sl=( slice(None), slice(0,2), ) )
        expected = np.array( [ 42283.62890625,  20401.44335938,  115423.2109375, ] )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_mask_default_and_other( self ):

        self.worldlines.data_masker.masks.append(
            {
                'data_key': 'NA',
                'data_min': np.nan,
                'data_max': np.nan,
                'mask': np.array([
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

    def test_get_masked_data_over_time( self ):

        # Mock data
        self.worldlines.data['A'] = np.array([
            [ 1., 2., 3., ],
            [ 4., 5., 6., ],
            [ 7., 8., 9, ],
            [ 10., 11., 12., ],
        ])
        self.worldlines.data['is_class_A'] = np.array([
            [ 0, 1, 0, ],
            [ 1, 1, 0, ],
            [ 0, 1, 1, ],
            [ 1, 0, 1, ],
        ]).astype( bool )
        
        # Mask data
        self.worldlines.data_masker.mask_data( 'T', 4500., np.inf )

        # Actual calculation
        actual = self.worldlines.get_masked_data_over_time(
            'A',
            snum = 550,
        )

        # Comparison
        expected = np.array([
            [ 1., 2., 3., ],
            [ 4., 5., 6., ],
        ])

        npt.assert_allclose( actual, expected )

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

########################################################################
########################################################################

class TestWorldlineDataKeyParser( unittest.TestCase ):

    def setUp( self ):

        self.key_parser = analyze_worldlines.WorldlineDataKeyParser()

    ########################################################################

    def test_basic( self ):

        actual, log_flag = self.key_parser.is_log_key( 'logBlah' )

        self.assertEqual( 'Blah', actual )

        assert log_flag

    ########################################################################

    def test_tiled( self ):

        actual, tiled_flag = self.key_parser.is_tiled_key( 'Blah_tiled' )

        self.assertEqual( 'Blah', actual )

        assert tiled_flag
