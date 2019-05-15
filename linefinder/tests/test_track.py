#!/usr/bin/env python
'''Testing for track.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
import inspect
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pytest
import shutil
import unittest

from linefinder import track
import linefinder.config as config
import galaxy_dive.read_data.snapshot as readsnap
import galaxy_dive.utils.constants as constants

########################################################################
# Global Setup
########################################################################

def example_fn_1( dfid, df ):

    dfid['above_med_Den'] = dfid['Den'] > np.nanmedian( df['Den'] )

    return dfid

def example_fn_2( dfid, df ):

    dfid['IDx2'] = dfid['ID'] * 2

    return dfid

def example_fn_3( dfid, df, id_finder ):

    dfid['snum'] = np.ones( len( dfid['ID'] ) ) * id_finder.snum

    return dfid

test_apply_fns = [ example_fn_1, example_fn_2 ]

default_data_p = {
    'sdir': './tests/data/test_data_with_new_id_scheme',
    'p_types': [0, ],
    'snum_start': 500,
    'snum_end': 600,
    'snum_step': 50,

    'out_dir': './tests/data/tracking_output',
    'tag': 'test',

    'custom_fns': test_apply_fns,
}

star_data_p = {
    'sdir': './tests/data/stars_included_test_data',
    'p_types': [0, 4],
    'snum_start': 500,
    'snum_end': 600,
    'snum_step': 50,

    'out_dir': './tests/data/tracking_output',
    'tag': 'test_star',

    'custom_fns': test_apply_fns,
}

early_star_data_p = {
    'sdir': './tests/data/stars_included_test_data',
    'p_types': [0, 4],
    'snum_start': 10,
    'snum_end': 11,
    'snum_step': 1,

    'out_dir': './tests/data/tracking_output',
    'tag': 'test_star_early',

    'custom_fns': test_apply_fns,
}

fire1_data_p = copy.deepcopy( star_data_p )
fire1_data_p['tag'] = 'test_fire1'

# Make sure that the ids sdir attribute resembles what would happen if it was generated.
for data_p in [ default_data_p, star_data_p, early_star_data_p, fire1_data_p, ]:
    id_filename = './tests/data/tracking_output/ids_{}.hdf5'.format( data_p['tag'] )
    ids_file = h5py.File( id_filename, 'a' )
    ids_file.attrs['sdir'] = os.path.abspath( data_p['sdir'] )
    ids_file.close()

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################

class TestConcatenateParticleData( unittest.TestCase ):

    def setUp( self ):

        self.id_finder = track.IDFinder()

        # The name of the function.
        self.fn = self.id_finder.concatenate_particle_data

    ########################################################################

    def test_basic( self ):
        '''Basically, does it work?'''

        # Input
        self.id_finder.sdir = './tests/data/test_data_with_new_id_scheme'
        self.id_finder.snum = 600
        self.id_finder.p_types = [0, ]
        self.id_finder.target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
        self.id_finder.target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])

        self.id_finder.concatenate_particle_data()
        actual = self.id_finder.full_snap_data

        expected = {
            'ID': np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235]),
            'ChildID': np.array([1945060136, 0, 0, 938428052, 893109954, 0]),
        }

        for key in expected.keys():
            npt.assert_allclose( actual[key], expected[key] )

    ########################################################################

    def test_works_for_stars( self ):

        # Input
        self.id_finder.sdir = './tests/data/test_data_with_new_id_scheme'
        self.id_finder.snum = 600
        self.id_finder.p_types = [0,4]
        self.id_finder.target_ids = np.array([24565150, 24079833, 13109563, 14147322, ])
        self.id_finder.target_child_ids = np.array([ 0, 0, 0, 0 ])

        self.id_finder.concatenate_particle_data()
        actual = self.id_finder.full_snap_data

        expected = {
            'ID': np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235,
                                              24565150, 24079833, 13109563, 14147322, 28067645, 10259537]),
            'ChildID': np.array([1945060136, 0, 0, 938428052, 893109954, 0,
                                                          0, 0, 0, 0, 0, 0]),
            }

        for key in expected.keys():
            npt.assert_allclose( actual[key], expected[key] )

########################################################################

class TestSelectIDs( unittest.TestCase ):

    def setUp(self):

        self.id_finder = track.IDFinder()

        # The name of the function.
        self.fn = self.id_finder.select_ids

    ########################################################################

    def test_runs(self):
        '''Test that it even runs'''

        # Dummy data set
        self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
        self.id_finder.full_snap_data = {
            'ID': np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
            'Den': np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
        }
        self.id_finder.full_snap_data['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        self.fn()

        assert True

    ########################################################################

    def test_works_simple( self ):
        '''Test that it runs in a simple case where there's no issues with duplicates.'''

        # Dummy data set
        self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
        self.id_finder.full_snap_data = {
            'ID': np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
            'Den': np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
        }
        self.id_finder.full_snap_data['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        dfid, df = self.fn()

        expected = {
            'ID': np.array([ 38913508, 3211791, 10952235 ]),
            'Den': np.array([ 3.45104532e-08,  6.80917722e-09, 1.54667816e-08 ]),
        }
        expected['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        for key in dfid.keys():
            npt.assert_allclose( dfid[key], expected[key] )

    ########################################################################

    def test_works_duplicates( self ):

        # Dummy data set
        self.id_finder.target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
        self.id_finder.target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])
        self.id_finder.full_snap_data = {
            'ID': np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235]),
            'ChildID': np.array([1945060136, 0, 0, 938428052, 893109954, 0]),
            'Den': np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
        }
        self.id_finder.full_snap_data['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        dfid, df = self.fn()

        expected = {
            'ID': self.id_finder.target_ids,
            'ChildID': self.id_finder.target_child_ids,
            'Den': np.array([ 3.45104532e-08, 3.80374093e-10, 6.80917722e-09, 1.54667816e-08 ]),
        }
        expected['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        for key in dfid.keys():
            npt.assert_allclose( dfid[key], expected[key] )

    ########################################################################

    def test_child_id_not_found( self ):
        '''Sometimes particles can stop existing. This occurs when the particle is a split particle,
        coming from some parent. In such a case, we want to stop tracking the particle. It's probably not worth it to
        track the parent, because that parent accumulated at least as much mass as itself in order to split,
        and as such we can't easily tell where that mass is coming from.
        '''

        self.id_finder.target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
        self.id_finder.target_child_ids = np.array([ 893109954, 15, 0, 0 ]) # The second ID here doesn't exist
        self.id_finder.full_snap_data = {
            'ID': np.array([36091289,  3211791, 41221636, 36091289, 36091289, 10952235]),
            'ChildID': np.array([1945060136, 0, 0, 938428052, 893109954, 0]),
            'Den': np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
        }
        self.id_finder.full_snap_data['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        dfid, df = self.fn()

        expected = {
            'ID': self.id_finder.target_ids,
            'ChildID': self.id_finder.target_child_ids,
            'Den': np.array([ 3.45104532e-08, np.nan, 6.80917722e-09, 1.54667816e-08 ]),
        }
        expected['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        for key in dfid.keys():
            npt.assert_allclose( dfid[key], expected[key] )


########################################################################

class TestApplyFunctions( unittest.TestCase ):

    def setUp(self):

        self.id_finder = track.IDFinder()

    ########################################################################

    def test_runs(self):
        '''Test that it even runs'''

        # Dummy data set
        self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
        self.id_finder.full_snap_data = {
            'ID': np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
            'Den': np.array([  3.80374093e-10,   6.80917722e-09,   3.02682572e-08, 1.07385445e-09,   3.45104532e-08,   1.54667816e-08]),
        }
        self.id_finder.full_snap_data['Den'] *= constants.UNITDENSITY_IN_NUMDEN

        dfid, df = self.id_finder.select_ids()

        dfid = self.id_finder.apply_functions( test_apply_fns, dfid, df )

        expected = {
            'above_med_Den': np.array([ True, False, True ]),
            'IDx2': np.array([ 77827016, 6423582, 21904470 ]),
        }
        for key in expected.keys():
            npt.assert_allclose( expected[key], dfid[key] )

    ########################################################################

    def test_uses_meta_data( self ):
        '''Test we can use things not directly stored in the data frame,
        like the snapshot number or the redshift.
        '''

        # Dummy data set
        self.id_finder.target_ids = np.array([ 38913508, 3211791, 10952235 ])
        self.id_finder.full_snap_data = {
            'ID': np.array([56037496,  3211791, 41221636, 63924292, 38913508, 10952235]),
        }
        self.id_finder.snum = 600

        dfid, df = self.id_finder.select_ids()

        dfid = self.id_finder.apply_functions( [ example_fn_3, ], dfid, df )

        expected = {
            'snum': np.array([ 600, 600, 600 ]),
        }
        for key in expected.keys():
            npt.assert_allclose( expected[key], dfid[key] )

########################################################################

class TestFindIds( unittest.TestCase ):

    def setUp( self ):

        self.id_finder = track.IDFinder()

        # The name of the function.
        self.fn = self.id_finder.find_ids

    ########################################################################

    def test_basic( self ):
        '''Basically, does it work?.'''

        # Input
        sdir = './tests/data/test_data_with_new_id_scheme'
        snum = 600
        p_types = [0, ]
        target_ids = np.array([ 36091289, 36091289, 3211791, 10952235 ])
        target_child_ids = np.array([ 893109954, 1945060136, 0, 0 ])

        # My knowledge, by hand
        target_inds = [4, 0, 1, 5]
        P = readsnap.readsnap( sdir, snum, 0, True, cosmological=True )

        expected = {
            'ID': target_ids,
            'ChildID': target_child_ids,
            'Den': np.array([ P['rho'][ind]*constants.UNITDENSITY_IN_NUMDEN for ind in target_inds ]),
        }

        dfid, redshift, attrs = self.fn( sdir, snum, p_types, target_ids, \
                                                            target_child_ids=target_child_ids)

        for key in expected.keys():
            npt.assert_allclose( dfid[key], expected[key] )

        # Make sure the redshift's right too
        npt.assert_allclose( redshift, 0. )

    ########################################################################

    def test_no_child_ids( self ):

        # Input
        sdir = './tests/data/stars_included_test_data'
        snum = 500
        p_types = [ 0, 4,]
        target_ids = np.array([ 24079833, 24565150, 14147322, ])
        target_child_ids = None

        # My knowledge, by hand
        target_inds = [ 1, 0, 5, ]
        target_ptype = [ 0, 0, 4, ]

        expected = {
            'ID': target_ids,
            'Den': np.array([ 3.12611002e-08,   2.98729116e-09, np.nan ])*constants.UNITDENSITY_IN_NUMDEN
        }

        dfid, redshift, attrs = self.fn( sdir, snum, p_types, target_ids, \
                                                            target_child_ids=target_child_ids)

        for key in expected.keys():
            npt.assert_allclose( dfid[key], expected[key] )

########################################################################

class TestSaveTargetedParticles( unittest.TestCase ):

    def setUp( self ):

        # Mock the code version so we don't repeatedly change test data
        patcher = patch( 'galaxy_dive.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        self.particle_tracker = track.ParticleTracker( **default_data_p )

        # The name of the function.
        self.fn = self.particle_tracker.save_particle_tracks

    ########################################################################

    def test_runs( self ):

        self.fn()

    ########################################################################

    def test_basic( self ):

        self.fn()

        f = h5py.File( 'tests/data/tracking_output/ptracks_test.hdf5', 'r' )

        expected_snum = np.arange(600, 490, -50)
        actual_snum = f['snum'][...]
        npt.assert_allclose( expected_snum, actual_snum )

        expected_rho_p0 =  np.array([  1.70068894e-08, 6.44416458e-08, 1.94556549e-09])*constants.UNITDENSITY_IN_NUMDEN
        actual_rho_p0 = f['Den'][...][0]
        npt.assert_allclose( expected_rho_p0, actual_rho_p0 )

        assert 'ChildID' in f.keys()

        assert 'Potential' in f.keys()

        expected_above_med =  np.array([ True, False, False, True ])
        actual_above_med = f['above_med_Den'][...][:,0]
        npt.assert_allclose( expected_above_med, actual_above_med )

    ########################################################################

    def test_works_with_stars( self ):

        self.particle_tracker = track.ParticleTracker( **star_data_p )
        self.particle_tracker.save_particle_tracks()

        f = h5py.File( 'tests/data/tracking_output/ptracks_test_star.hdf5', 'r' )

        expected_snum = np.arange(600, 490, -50)
        actual_snum = f['snum'][...]
        npt.assert_allclose( expected_snum, actual_snum )

        expected_id = np.array([24565150, 24079833, 13109563, 14147322])
        actual_id = f['ID'][...]
        npt.assert_allclose( expected_id, actual_id )

        expected_rho_500 = np.array([
            2.98729116e-09,
            3.12611002e-08,
            8.95081308e-04,
            np.nan,
        ])*constants.UNITDENSITY_IN_NUMDEN
        actual_rho_500 = f['Den'][...][:,-1]
        npt.assert_allclose( expected_rho_500, actual_rho_500 )

        expected_ptype_p0 = np.array([ 4, 0, 0 ])
        actual_ptype_p0 = f['PType'][...][0]
        npt.assert_allclose( expected_ptype_p0, actual_ptype_p0 )

        assert 'ChildID' in f.keys()

    ########################################################################

    def test_works_with_stars_early_on( self ):
        '''Test this works even in early snapshots, when there are no stars.'''

        self.particle_tracker = track.ParticleTracker( **early_star_data_p )
        self.particle_tracker.save_particle_tracks()

        f = h5py.File( 'tests/data/tracking_output/ptracks_test_star_early.hdf5', 'r' )

        expected_snum = np.array([ 11, 10 ])
        actual_snum = f['snum'][...]
        npt.assert_allclose( expected_snum, actual_snum )

        expected_id = np.array([2040268, 7909745, 8961984])
        actual_id = f['ID'][...]
        npt.assert_allclose( expected_id, actual_id )

    ########################################################################

    def test_works_fire1( self ):

        self.particle_tracker = track.ParticleTracker( **fire1_data_p )
        self.particle_tracker.save_particle_tracks()

        f = h5py.File( 'tests/data/tracking_output/ptracks_test_fire1.hdf5', 'r' )

        expected_snum = np.arange(600, 490, -50)
        actual_snum = f['snum'][...]
        npt.assert_allclose( expected_snum, actual_snum )

        expected_id = np.array([24565150, 24079833, 13109563, 14147322])
        actual_id = f['ID'][...]
        npt.assert_allclose( expected_id, actual_id )

        expected_rho_500 = np.array([ 2.98729116e-09,   3.12611002e-08,   8.95081308e-04, np.nan ])*constants.UNITDENSITY_IN_NUMDEN
        actual_rho_500 = f['Den'][...][:,-1]
        npt.assert_allclose( expected_rho_500, actual_rho_500 )

        expected_ptype_p0 = np.array([ 4, 0, 0 ])
        actual_ptype_p0 = f['PType'][...][0]
        npt.assert_allclose( expected_ptype_p0, actual_ptype_p0 )

    ########################################################################

    def test_has_attributes( self ):

        self.fn()

        f = h5py.File( 'tests/data/tracking_output/ptracks_test.hdf5', 'r' )

        # Load one of the original snapshots to compare
        P = readsnap.readsnap( 'tests/data/test_data_with_new_id_scheme', 600, 0, True, cosmological=True )

        compare_keys = [ 'omega_matter', 'omega_lambda', 'hubble' ]

        for key in compare_keys:
            npt.assert_allclose( P[key], f.attrs[key] )

        for key in default_data_p.keys():
            if key == 'check_same_sdir' or key == 'custom_fns':
                continue
            elif key == 'custom_fns_str':
                for i, fn in enumerate( default_data_p[key] ):
                    assert inspect.getsource( fn ) == f['parameters'].attrs[key][i]
            else:
                assert default_data_p[key] == f['parameters'].attrs[key]

    ########################################################################

    def test_get_target_ids( self ):

        # Remove any interfering attributes
        if hasattr( self.particle_tracker, 'target_ids'):
            del self.particle_tracker.target_ids

        self.particle_tracker.get_target_ids()

        expected = {
            'target_ids': np.array([ 36091289, 36091289, 3211791, 10952235 ]),
            'target_child_ids': np.array([ 893109954, 1945060136, 0, 0 ]),
        }

        for key in expected.keys():
            npt.assert_allclose( getattr( self.particle_tracker, key ), expected[key] )

########################################################################
########################################################################

class TestSaveTargetedParticlesParallel( unittest.TestCase ):

    def setUp( self ):

        # Mock the code version so we don't repeatedly change test data
        patcher = patch( 'galaxy_dive.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        kwargs = dict( default_data_p )
        kwargs['n_processors'] = 2

        self.particle_tracker = track.ParticleTracker( **kwargs )

        # The name of the function.
        self.fn = self.particle_tracker.save_particle_tracks

    ########################################################################

    def test_basic( self ):

        self.fn()

        f = h5py.File( 'tests/data/tracking_output/ptracks_test.hdf5', 'r' )

        expected_snum = np.arange(600, 490, -50)
        actual_snum = f['snum'][...]
        npt.assert_allclose( expected_snum, actual_snum )

        expected_rho_p0 =  np.array([  1.70068894e-08, 6.44416458e-08, 1.94556549e-09])*constants.UNITDENSITY_IN_NUMDEN
        actual_rho_p0 = f['Den'][...][0]
        npt.assert_allclose( expected_rho_p0, actual_rho_p0 )

        assert 'ChildID' in f.keys()

########################################################################

class TestSaveTargetedParticlesJug( unittest.TestCase ):

    def setUp( self ):

        # Switch dirs because executables can't be run while in the package
        os.chdir( '..' )

        # Mock the code version so we don't repeatedly change test data
        patcher = patch( 'galaxy_dive.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        self.savefile = './linefinder/tests/data/tracking_output/ptracks_test_jug.hdf5'

        if os.path.isfile( self.savefile ):
            os.system( 'rm {}'.format( self.savefile ) )

    ########################################################################

    def tearDown( self ):

        # Switch back so we don't mess up other tests
        os.chdir( 'linefinder' )

        jugdata_dir = './tests/track_jugfile.jugdata'

        if os.path.isdir( jugdata_dir ):
            shutil.rmtree( jugdata_dir )

    ########################################################################

    @slow
    def test_basic( self ):

        os.system( "{} ./linefinder/tests/track_jugfile.py &".format(
            config.JUG_EXEC_PATH )
        )
        os.system( "{} ./linefinder/tests/track_jugfile.py".format(
            config.JUG_EXEC_PATH )
        )

        f = h5py.File( './linefinder/tests/data/tracking_output/ptracks_test_jug.hdf5', 'r' )

        expected_snum = np.arange(600, 490, -50)
        actual_snum = f['snum'][...]
        npt.assert_allclose( expected_snum, actual_snum )

        expected_rho_p0 =  np.array([  1.70068894e-08, 6.44416458e-08, 1.94556549e-09])*constants.UNITDENSITY_IN_NUMDEN
        actual_rho_p0 = f['Den'][...][0]
        npt.assert_allclose( expected_rho_p0, actual_rho_p0 )

        assert 'ChildID' in f.keys()
