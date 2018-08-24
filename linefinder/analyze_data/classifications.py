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


class Classifications( object ):
    '''Loads and analyzes data created by classify.py
    '''

    def __init__( self, data_dir, tag ):
        '''
        Args:
            data_dir (str) : Data directory for the classifications data
            tag (str) : Identifying tag for the data to load.
        '''

        # Store the arguments
        for arg in locals().keys():
            setattr( self, arg, locals()[arg] )

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
                self.data_attrs[key] = f.attrs[key]

            # Store the parameters
            self.parameters = {}
            param_grp = f['parameters']
            for key in param_grp.attrs.keys():
                self.parameters[key] = param_grp.attrs[key]

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

    ########################################################################

    def calc_base_fractions( self, return_or_store='return', mask=None ):
        '''Get first order results of the classifications data, in the form of
        fractions of each main type of classifications data, relative to the
        total.

        Args:
            return_or_store (str) :
                Whether to return base_fractions or store
                it as an attribute

            mask (np.array of bools) :
                What mask to apply to the data, if any

        Returns:
            base_fractions (dict) :
                Contains the fractions for each main type of classifications
                data, evaluated at the final snapshot
        '''

        base_fractions = {}

        # Get the total number of particles
        is_pristine = self.get_data( 'is_pristine', mask=mask )
        n_classified = float( is_pristine.size )

        base_fractions['fresh accretion'] = \
            float( is_pristine.sum() ) / n_classified
        base_fractions['merger'] = \
            float( self.get_data( 'is_merger', mask=mask ).sum() ) / \
            n_classified
        base_fractions['intergalactic transfer'] = \
            float( self.get_data( 'is_mass_transfer', mask=mask ).sum() ) / \
            n_classified

        # For wind, only evaluate at the last snapshot
        base_fractions['wind'] = \
            float( self.get_data( 'is_wind', mask=mask, slice_index=0 ).sum() )\
            / n_classified

        if return_or_store == 'return':
            return base_fractions
        else:
            self.base_fractions = base_fractions
