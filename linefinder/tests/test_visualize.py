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

from linefinder import visualize
import linefinder.config as config

########################################################################
# Global Setup
########################################################################

firefly_dir = './tests/data/firefly'

########################################################################

class TestExportToFirefly( unittest.TestCase ):

    def setUp( self ):

        self.default_kwargs = {
            'data_dir': './tests/data/tracking_output_for_analysis',
            'tag': 'analyze',
            'halo_data_dir': './tests/data/ahf_test_data',

            'install_firefly': True,

            'export_to_firefly_kwargs': {
                'firefly_dir': firefly_dir,
                'disk_radius': ( 0.1, 'Rvir' ),
            },
        }
        halo_tracks_fps = [
            'halo_tracks_snum600.hdf5',
            'halo_tracks_pathlines.hdf5'
        ]
        self.halo_tracks_fps = [
            os.path.join( self.default_kwargs['halo_data_dir'], _ )
            for _ in halo_tracks_fps
        ]

        # Start with clean dirs
        if os.path.isdir( firefly_dir ):
            shutil.rmtree( firefly_dir )

    def tearDown( self ):

        # End with clean dirs
        for halo_tracks_fp in self.halo_tracks_fps:
            if os.path.isfile( halo_tracks_fp ):
                os.remove( halo_tracks_fp )

        if os.path.isdir( firefly_dir ):
            shutil.rmtree( firefly_dir )

    ########################################################################

    def test_basic( self ):
        '''Basically, does it run?'''

        visualize.export_to_firefly(
            **self.default_kwargs,
        )

        data_dir = os.path.join( firefly_dir, 'src', 'Firefly', 'static', 'data', 'analyze_pathlines' )
        assert os.path.isdir( data_dir )
        assert os.path.isfile( os.path.join( data_dir, 'DataAll000.json' ) )

    ########################################################################

    def test_halo_tracks( self ):

        self.default_kwargs['export_to_firefly_kwargs']['include_halo_tracks'] = True
        
        visualize.export_to_firefly(
            **self.default_kwargs,
        )

        for halo_track_fp in self.halo_tracks_fps:
            f = h5py.File( halo_track_fp, 'r' )
            fields = [
                'coordinates',
                'tracked_arrays',
            ]
            for field in fields:
                assert field in f.keys()
