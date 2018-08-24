'''Testing for data_management.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import linefinder.utils.trove_management as p_trove_management
import linefinder.utils.file_management as file_management

########################################################################

file_format = 'galids_{}_snum{}.hdf5'
sim_names = [ 'm12i', 'm11q', ]
snums = [ 500, 550, 600, ]

########################################################################
########################################################################

class TestPathfinderTroveManager( unittest.TestCase ):

    def setUp( self ):

        self.p_trove_manager = p_trove_management.PathfinderTroveManager(
            file_format,
            sim_names,
            snums,
        )

    ########################################################################

    def test_combinations( self ):

        actual = self.p_trove_manager.combinations

        expected = [
            ( 'm12i', 500 ),
            ( 'm12i', 550 ),
            ( 'm12i', 600 ),
            ( 'm11q', 500 ),
            ( 'm11q', 550 ),
            ( 'm11q', 600 ),
        ]

        self.assertEqual( expected, actual )

    ########################################################################

    def test_get_file( self ):

        actual = self.p_trove_manager.get_file( 'm12i', 500 )

        file_manager = file_management.FileManager()
        expected = os.path.join(
            file_manager.get_linefinder_dir( 'm12i' ),
            'galids_m12i_snum500.hdf5',
        )
            
        self.assertEqual( expected, actual )

