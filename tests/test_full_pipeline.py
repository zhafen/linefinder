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
outdir = './tests/data/tracking_output_for_analysis'
tag = 'analyze'

selector_kwargs = {
    'snum_start': snap_ini,
    'snum_end': snap_end,
    'snum_step': snap_step,

    'out_dir': outdir,
    'tag': tag,

    'p_types': types,

    'snapshot_kwargs': {
        'sdir': sdir,
        'load_additional_ids': False,
        'ahf_index': mtree_halos_index,
        'analysis_dir': ahf_sdir,
    }
}

sampler_kwargs = {
    'out_dir': outdir,
    'tag': tag,
}

# Tracking Parameters
tracker_kwargs = {
    'out_dir': outdir,
    'tag': tag,
}

# Galaxy Finding Parameters
gal_finder_kwargs = {
    'out_dir': outdir,
    'tag': tag,

    'ahf_data_dir': ahf_sdir,
    'main_mt_halo_id': 0,

    'n_processors': 1,

    'length_scale': 'Rvir',
}

# Classifying Parameters
classifier_kwargs = {
    'out_dir': outdir,
    'tag': tag,

    'velocity_scale': 'Vc(Rvir)',
}

ptracks_filename = os.path.join( outdir, 'ptracks_analyze.hdf5' )
galids_filename = os.path.join( outdir, 'galids_analyze.hdf5' )
classifications_filename = os.path.join( outdir, 'classifications_analyze.hdf5' )
events_filename = os.path.join( outdir, 'events_analyze.hdf5' )

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################
########################################################################


class TestFullWorldline( unittest.TestCase ):
    '''These are really integration tests.'''

    def setUp( self ):

        for filename in [ ptracks_filename, galids_filename, classifications_filename, events_filename ]:
            if os.path.isfile( filename ):
                os.remove( filename )

    ########################################################################

    @slow
    def test_full_pipeline( self ):
        '''Except the id selecting... This makes sure the full pipeline just runs.'''

        pathfinder.run_pathfinder(
            tracker_kwargs = tracker_kwargs,
            gal_finder_kwargs = gal_finder_kwargs,
            classifier_kwargs = classifier_kwargs,
            run_id_selection = False,
            run_id_sampling = False,
        )

    ########################################################################

    @slow
    def test_actually_full_pipeline( self ):
        '''Test that everything runs, including ID selecting.'''

        pathfinder.run_pathfinder(
            selector_kwargs = selector_kwargs,
            sampler_kwargs = sampler_kwargs,
            tracker_kwargs = tracker_kwargs,
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

        assert False, "Need to test that everything matches."

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
