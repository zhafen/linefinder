#!/usr/bin/env python
'''Code for managing data troves.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import itertools
import os

import galaxy_dive.data_management.trove_management as trove_management

import linefinder.utils.file_management as file_management

########################################################################
########################################################################


class LinefinderTroveManager( trove_management.TroveManager ):
    '''Class for managing troves of data.'''

    ########################################################################

    def get_file( self, *args ):
        '''Default method for getting the data filename.
        
        Args:
            *args :
                Arguments provided. Assumes args[0] is the data dir.

        Returns:
            Filename for a given combination of args.
        '''

        file_manager = file_management.FileManager()

        data_dir = file_manager.get_linefinder_dir( args[0] )

        filename = self.file_format.format( *args )

        return os.path.join( data_dir, filename )

