#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
from mock import patch
import numpy.testing as npt
import os
import pytest
import shutil
import unittest

import galaxy_diver.galaxy_finder.finder as galaxy_finder
from pathfinder import galaxy_find

import pathfinder.config as config

########################################################################
# Useful global test variables
########################################################################

gal_finder_kwargs = {
    'length_scale': 'Rvir',

    'redshift': 0.16946003,
    'snum': 500,
    'hubble': 0.70199999999999996,
    'ahf_data_dir': './tests/data/ahf_test_data',
    'mtree_halos_index': 600,
    'main_mt_halo_id': 0,
    'halo_file_tag': 'smooth',

    'galaxy_cut': 0.1,
    'ids_to_return': [
        'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id',
        'mt_halo_id', 'mt_gal_id', 'd_gal', 'd_other_gal_scaled', ],
    'minimum_criteria': 'n_star',
    'minimum_value': 0,
}

ptrack_gal_finder_kwargs = {
    'length_scale': 'Rvir',
    'ids_to_return': [
        'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id',
        'mt_halo_id', 'mt_gal_id', 'd_gal', 'd_other_gal_scaled', ],
    'minimum_criteria': 'n_star',
    'minimum_value': 0,

    'galaxy_cut': 0.1,

    'ahf_data_dir': './tests/data/ahf_test_data',
    'out_dir': './tests/data/tracking_output',
    'tag': 'test',
    'mtree_halos_index': 600,
    'main_mt_halo_id': 0,
}

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################


class TestParticleTrackGalaxyFinder( unittest.TestCase ):

    def setUp( self ):

        # Mock the code version so we don't repeatedly change test data
        patcher = patch( 'galaxy_diver.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        self.originalfile = './tests/data/tracking_output/ptracks_test.hdf5'
        self.savefile = './tests/data/tracking_output/galids_test.hdf5'

        if os.path.isfile( self.savefile ):
            os.system( 'rm {}'.format( self.savefile ) )

    ########################################################################

    def test_find_galaxies_for_particle_tracks( self ):

        particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder(
            **ptrack_gal_finder_kwargs )
        particle_track_gal_finder.find_galaxies_for_particle_tracks()

        f = h5py.File( self.savefile, 'r' )
        g = h5py.File( self.originalfile, 'r' )

        # What we expect (calculated using the positions of the particles in
        # snum 500 )
        particle_positions = g['P'][...][:, -1]
        gal_finder = galaxy_finder.GalaxyFinder(
            particle_positions, **gal_finder_kwargs )
        expected = gal_finder.find_ids()

        # Make the comparison
        for key in expected.keys():
            npt.assert_allclose(
                expected[key], f[key][...][:, -1],
                err_msg='Key {} failed'.format( key ), rtol=1e-4 )

        # What we expect (calculated using the positions of the particles in
        # snum 550 )
        gal_finder_kwargs_copy = dict( gal_finder_kwargs )
        gal_finder_kwargs_copy['snum'] = g['snum'][1]
        gal_finder_kwargs_copy['redshift'] = g['redshift'][1]
        particle_positions = g['P'][...][:, 1]
        gal_finder = galaxy_finder.GalaxyFinder(
            particle_positions, **gal_finder_kwargs_copy )

        # Make the comparison
        expected = gal_finder.find_ids()
        for key in expected.keys():
            npt.assert_allclose(
                expected[key],
                f[key][...][:, 1],
                err_msg="Key '{}' failed".format( key ),
                rtol=1e-4
            )

        # Make sure the main MT halo ID is the one we expect.
        assert f.attrs['main_mt_halo_id'] == 0

        # Make sure we have stored the data parameters too.
        for key in ptrack_gal_finder_kwargs.keys():
            if (key != 'ids_to_return') and (key != 'main_mt_halo_id'):
                assert \
                    ptrack_gal_finder_kwargs[key] == f['parameters'].attrs[key]

########################################################################


class TestParticleTrackGalaxyFinderParallel( unittest.TestCase ):

    def setUp( self ):

        # Mock the code version so we don't repeatedly change test data
        patcher = patch( 'galaxy_diver.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        self.originalfile = './tests/data/tracking_output/ptracks_test.hdf5'
        self.savefile = './tests/data/tracking_output/galids_test_parallel.hdf5'

        if os.path.isfile( self.savefile ):
            os.system( 'rm {}'.format( self.savefile ) )

    ########################################################################

    def test_find_galaxies_for_particle_tracks_parallel( self ):

        parallel_kwargs = dict( ptrack_gal_finder_kwargs )
        parallel_kwargs['ptracks_tag'] = 'test'
        parallel_kwargs['tag'] = 'test_parallel'
        parallel_kwargs['n_processors'] = 2

        particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder(
            **parallel_kwargs )
        particle_track_gal_finder.find_galaxies_for_particle_tracks()

        expected = \
            h5py.File( './tests/data/tracking_output/galids_test.hdf5', 'r' )
        actual = h5py.File( self.savefile, 'r' )

        for key in expected.keys():
            if key != 'parameters':
                npt.assert_allclose( expected[key], actual[key] )

########################################################################


class TestParticleTrackGalaxyFinderJug( unittest.TestCase ):

    def setUp( self ):

        # Mock the code version so we don't repeatedly change test data
        patcher = patch( 'galaxy_diver.utils.utilities.get_code_version' )
        self.addCleanup( patcher.stop )
        self.mock_code_version = patcher.start()

        self.originalfile = './tests/data/tracking_output/ptracks_test.hdf5'
        self.savefile = './tests/data/tracking_output/galids_test_jug.hdf5'

        jugdata_dir = './tests/find_galaxies_for_ptracks_jugfile.jugdata'

        if os.path.isfile( self.savefile ):
            os.system( 'rm {}'.format( self.savefile ) )
        if os.path.isdir( jugdata_dir ):
            shutil.rmtree( jugdata_dir )

    ########################################################################

    @slow
    def test_find_galaxies_for_particle_tracks_parallel( self ):

        os.system( "{} ./tests/find_galaxies_for_ptracks_jugfile.py &".format(
            config.JUG_EXEC_PATH )
        )
        os.system( "{} ./tests/find_galaxies_for_ptracks_jugfile.py".format(
            config.JUG_EXEC_PATH )
        )

        expected = \
            h5py.File( './tests/data/tracking_output/galids_test.hdf5', 'r' )
        actual = h5py.File( self.savefile, 'r' )

        for key in expected.keys():
            if key != 'parameters':
                npt.assert_allclose( expected[key], actual[key] )
