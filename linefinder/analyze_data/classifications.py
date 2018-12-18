#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os

import galaxy_dive.utils.utilities as utilities

########################################################################
########################################################################


class Classifications( object ):
    '''Loads and analyzes data created by classify.py
    '''

    @utilities.store_parameters
    def __init__( self, data_dir, tag ):
        '''
        Args:
            data_dir (str) : Data directory for the classifications data
            tag (str) : Identifying tag for the data to load.
        '''

        # Open the file
        classifications_filepath = os.path.join(
            data_dir, 'classifications_{}.hdf5'.format( tag ) )
        with h5py.File( classifications_filepath, 'r' ) as f:

            # Store the data
            self.data = {}
            for key in f.keys():
                if key != 'parameters':
                    self.data[key] = f[key][...]

            # Store the data attributes
            self.data_attrs = {}
            for key in f.attrs.keys():
                self.data_attrs[key] = utilities.check_and_decode_bytes(
                    f.attrs[key]
                )

            # Store the parameters
            self.parameters = {}
            param_grp = f['parameters']
            for key in param_grp.attrs.keys():
                self.parameters[key] = utilities.check_and_decode_bytes(
                    param_grp.attrs[key]
                )

    ########################################################################

    def get_data( self, data_key, mask=None, slice_index=None ):
        '''Get the data from the data dictionary. Useful (over just accessing
        the array) when applying additional functions onto it.

        Args:
            data_key (str) :
                Key for the relevant data.

            mask (np.array of bools) :
                What mask to apply to the data, if any

            slice_index (int) :
                If getting only a particular slice (for the two
                dimensional arrays like 'is_wind), what slice?

        Returns:
            data_arr (np.array) : Array of the requested data
        '''

        data_arr = self.data[data_key]

        if slice_index is not None:
            data_arr = data_arr[:, slice_index]

        if mask is not None:
            data_arr = np.ma.masked_array( data_arr, mask=mask ).compressed()

        return data_arr

