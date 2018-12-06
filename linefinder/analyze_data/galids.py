#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os

########################################################################
########################################################################

class GalIDs( object ):
    '''Loads and analyzes data created by galaxy_link.py
    '''

    def __init__( self, data_dir, tag ):
        '''
        Args:
            data_dir (str) : Data directory for the classified data
            tag (str) : Identifying tag for the data to load.
        '''

        # Store the arguments
        for arg in locals().keys():
            setattr( self, arg, locals()[arg] )

        # Open the file
        galids_filepath = os.path.join( data_dir, 'galids_{}.hdf5'.format( tag ) )
        with h5py.File( galids_filepath, 'r' ) as f:

            # Store the data
            self.data = {}
            for key in f.keys():
                if key != 'parameters':
                    self.data[key] = f[key][...]

            # Store the data attributes
            self.data_attrs = {}
            for key in f.attrs.keys():
                self.data_attrs[key] = f.attrs[key]

            # Store the parameters
            self.parameters = {}
            param_grp = f['parameters']
            for key in param_grp.attrs.keys():
                self.parameters[key] = param_grp.attrs[key]

