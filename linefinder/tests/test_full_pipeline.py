#!/usr/bin/env python
'''Testing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import numpy.testing as npt
import os
import pytest
import unittest

import linefinder.config as config
import linefinder.linefinder as linefinder

########################################################################
# Global variables

# Information about the input data
sdir = './tests/data/test_data_with_new_id_scheme'
ahf_sdir = './tests/data/ahf_test_data'
types = [ 0, 4, ]
snap_ini = 500
snap_end = 600
snap_step = 50
# By default, we assume that we've run AHF on every snapshot (we better have),
#   and that we're running tracking on all snapshots
mtree_halos_index = snap_end

# Information about what the output data should be called.
out_dir = './tests/data/tracking_output_for_analysis'
out_dir2 = './tests/data/full_linefinder_output'
out_dir3 = './linefinder/tests/data/full_linefinder_output'
tag = 'analyze'

selector_kwargs = {
    'snum_start': snap_ini,
    'snum_end': snap_end,
    'snum_step': snap_step,

    'p_types': types,

    'snapshot_kwargs': {
        'sdir': sdir,
        'load_additional_ids': True,
        'ahf_index': mtree_halos_index,
        'analysis_dir': ahf_sdir,
    }
}

sampler_kwargs = {
    'n_samples': 2,
}

# Tracking Parameters
tracker_kwargs = {
}

# Galaxy Finding Parameters
gal_linker_kwargs = {
    'halo_data_dir': ahf_sdir,
    'main_mt_halo_id': 0,

    'n_processors': 1,

    'length_scale': 'Rvir',
    'mt_length_scale': 'Rvir',
}

# Classifying Parameters
classifier_kwargs = {
    'velocity_scale': 'Vc(Rvir)',
}

ids_full_filename = os.path.join( out_dir, 'ids_full_analyze.hdf5' )
ids_filename = os.path.join( out_dir, 'ids_analyze.hdf5' )
ptracks_filename = os.path.join( out_dir, 'ptracks_analyze.hdf5' )
galids_filename = os.path.join( out_dir, 'galids_analyze.hdf5' )
classifications_filename = os.path.join( out_dir, 'classifications_analyze.hdf5' )
events_filename = os.path.join( out_dir, 'events_analyze.hdf5' )

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################
########################################################################


class TestLinefinderPartial( unittest.TestCase ):
    '''These are really integration tests.'''

    def setUp( self ):

        file_set = [
            'ptracks_analyze.hdf5',
            'galids_analyze.hdf5',
            'classifications_analyze.hdf5',
            'events_analyze.hdf5',
        ]
        for filename in file_set:

            full_filename = os.path.join( out_dir, filename )

            if os.path.isfile( full_filename ):
                os.remove( full_filename )

    ########################################################################

    @slow
    def test_pipeline( self ):
        '''Except the id selecting... This makes sure the full pipeline just runs.'''

        linefinder.run_linefinder(
            out_dir = out_dir,
            tag = 'analyze',
            tracker_kwargs = tracker_kwargs,
            gal_linker_kwargs = gal_linker_kwargs,
            classifier_kwargs = classifier_kwargs,
            run_id_selecting = False,
            run_id_sampling = False,
        )

########################################################################
########################################################################

def get_file_set( tag ):
    '''Get the names of all the files produced.'''

    file_set = [
        'ids_full_{}.hdf5'.format( tag ),
        'ids_{}.hdf5'.format( tag ),
        'ptracks_{}.hdf5'.format( tag ),
        'galids_{}.hdf5'.format( tag ),
        'classifications_{}.hdf5'.format( tag ),
        'events_{}.hdf5'.format( tag ),
    ]

    return file_set

class TestLinefinder( unittest.TestCase ):
    '''These are really integration tests.'''

    def setUp( self ):

        # Delete any pre-existing files
        for filename in get_file_set( 'analyze' ):
            full_filename = os.path.join( out_dir2, filename )
            if os.path.isfile( full_filename ):
                os.remove( full_filename )

    ########################################################################

    def tearDown( self ):

        #os.system( "rm -r ./tests/*jugdata" )
        # os.system( "rm -r ./tests/data/full_linefinder_output/*" )
        pass

    ########################################################################

    @slow
    def test_full_pipeline( self ):
        '''Test that everything runs, including ID selecting.'''

        # Choose the same seed for reproducibility.
        # Also, when I was creating the test data I wasn't careful enough to
        # Make sure that the sub-sampled snapshots had always consistent IDs...
        np.random.seed( 1234 )

        linefinder.run_linefinder(
            out_dir = out_dir2,
            tag = tag,
            selector_kwargs = selector_kwargs,
            sampler_kwargs = sampler_kwargs,
            gal_linker_kwargs = gal_linker_kwargs,
            classifier_kwargs = classifier_kwargs,
        )

########################################################################

class TestLinefinderJug( unittest.TestCase ):

    def setUp( self ):

        # We cannot execute scripts from the package dir
        os.chdir( '..' )

        # Delete any pre-existing files
        for filename in get_file_set( 'jug' ):
            full_filename = os.path.join( out_dir3, filename )
            if os.path.isfile( full_filename ):
                os.remove( full_filename )

    ########################################################################

    def tearDown( self ):

        # Switch back so we don't mess up other tests
        os.chdir( 'linefinder' )

        os.system( "rm -r ./tests/data/full_linefinder_output/jug.jugdata" )

    ########################################################################

    @slow
    def test_full_pipeline_jug( self ):
        '''Make sure everything runs and matches, including ID selecting.'''

        # Choose the same seed for reproducibility.
        # Also, when I was creating the test data I wasn't careful enough to
        # Make sure that the sub-sampled snapshots had always consistent IDs...
        np.random.seed( 1234 )

        os.system( "{} ./linefinder/tests/linefinder_jugfile.py &".format(
            config.JUG_EXEC_PATH )
        )
        os.system( "{} ./linefinder/tests/linefinder_jugfile.py".format(
            config.JUG_EXEC_PATH )
        )

        # TODO: Fix this.  I can't actually do this test right now, because jug
        # the traditional method aren't sampling the same particles...
        # for non_jug, jug in zip(
        #     get_file_set( 'analyze' ),
        #     get_file_set( 'jug' ),
        # ):

        #     non_jug_filename = os.path.join( out_dir3, non_jug )
        #     jug_filename = os.path.join( out_dir3, jug )

        #     non_jug_f = h5py.File( non_jug_filename, 'r' )
        #     jug_f = h5py.File( jug_filename, 'r' )

        #     # Do the actual check.
        #     for key in non_jug_f.keys():

        #         # Skip over this one, since it can't be compared.
        #         if key == 'parameters':
        #             continue

        #         npt.assert_allclose( non_jug_f[key][...], jug_f[key][...] )

        # At least check that the files exist and can be opened...
        for jug in get_file_set( 'jug' ):
            jug_filename = os.path.join( out_dir3, jug )
            jug_f = h5py.File( jug_filename, 'r' )

########################################################################
########################################################################


class TestCreateAnalysisData( unittest.TestCase ):
    '''Strictly speaking, these aren't really tests, so much as a kind of hacky way to generate test data for
    testing the analysis tools. Of course, if they break then something's wrong.'''

    @slow
    def test_create_classification_data( self ):

        f = h5py.File( classifications_filename, 'a' )

        for key in [ 'is_mass_transfer', 'is_merger', 'is_preprocessed', 'is_pristine', 'is_wind' ]:
            del f[key]

        f['is_mass_transfer'] = np.array( [ 0, 1, 0, 0, ] ).astype( bool )
        f['is_merger'] = np.array( [ 0, 0, 1, 1, ] ).astype( bool )
        f['is_preprocessed'] = np.array( [ 0, 1, 1, 1, ] ).astype( bool )
        f['is_pristine'] = np.array( [ 1, 0, 0, 0, ] ).astype( bool )
        f['is_wind'] = np.array( [
            [ 0, 0, 0, ],
            [ 1, 1, 0, ],
            [ 1, 1, 1, ],
            [ 0, 0, 0, ],
        ] ).astype( bool )
        f.close()
