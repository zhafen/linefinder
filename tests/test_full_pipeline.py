#!/usr/bin/env python
'''Testing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os
import pytest
import unittest

import pathfinder.config as config
import pathfinder.pathfinder as pathfinder

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
out_dir2 = './tests/data/full_pathfinder_output'
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
gal_finder_kwargs = {
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


class TestPathfinderPartial( unittest.TestCase ):
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

        pathfinder.run_pathfinder(
            out_dir = out_dir,
            tag = 'analyze',
            tracker_kwargs = tracker_kwargs,
            gal_finder_kwargs = gal_finder_kwargs,
            classifier_kwargs = classifier_kwargs,
            run_id_selecting = False,
            run_id_sampling = False,
        )

########################################################################
########################################################################


class TestPathfinder( unittest.TestCase ):
    '''These are really integration tests.'''

    def setUp( self ):

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

        # Delete any pre-existing files
        file_sets = [ get_file_set( 'analyze' ), get_file_set( 'jug' ) ]
        for file_set_ in file_sets:
            for filename in file_set_:
                full_filename = os.path.join( out_dir2, filename )
                if os.path.isfile( full_filename ):
                    os.remove( full_filename )

    ########################################################################

    def tearDown( self ):

        #os.system( "rm -r ./tests/*jugdata" )
        os.system( "rm -r ./tests/data/full_pathfinder_output/*" )

    ########################################################################

    @slow
    def test_full_pipeline( self ):
        '''Test that everything runs, including ID selecting.'''

        pathfinder.run_pathfinder(
            out_dir = out_dir2,
            tag = tag,
            selector_kwargs = selector_kwargs,
            sampler_kwargs = sampler_kwargs,
            gal_finder_kwargs = gal_finder_kwargs,
            classifier_kwargs = classifier_kwargs,
        )

    ########################################################################

    @slow
    def test_full_pipeline_jug( self ):
        '''Make sure everything runs and matches, including ID selecting.'''

        os.system( "{} ./tests/pathfinder_jugfile.py &".format(
            config.JUG_EXEC_PATH )
        )
        os.system( "{} ./tests/pathfinder_jugfile.py".format(
            config.JUG_EXEC_PATH )
        )

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
