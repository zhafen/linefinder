#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import os

import galaxy_dive.utils.utilities as utilities

########################################################################
########################################################################


class IDs( object ):
    '''Loads and analyzes data created by select.py
    '''

    @utilities.store_parameters
    def __init__( self, data_dir, tag ):
        '''
        Args:
            data_dir (str) : Data directory for the classified data
            tag (str) : Identifying tag for the data to load.
        '''

        # Open the file
        ids_filepath = os.path.join( data_dir, 'ids_{}.hdf5'.format( tag ) )
        with h5py.File( ids_filepath, 'r' ) as f:

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
                self.parameters[key] = utilities.check_and_decode_bytes(
                    param_grp.attrs[key]
                )
                    

            # Store the parameters
            self.snapshot_parameters = {}
            snap_param_grp = f['parameters/snapshot_parameters']
            for key in snap_param_grp.attrs.keys():
                self.snapshot_parameters[key] = utilities.check_and_decode_bytes(
                     snap_param_grp.attrs[key]
                )

            # Store the used data filters
            self.data_filters = {}
            try:
                filters_grp = f['parameters/data_filters']
                for filters_subset in filters_grp.keys():
                    subgroup = filters_grp[filters_subset]
                    self.data_filters[filters_subset] = {}
                    for key in subgroup.attrs.keys():
                        self.data_filters[filters_subset][key] = \
                            subgroup.attrs[key]
            except KeyError:
                print( "Failed to load data_filters. Using older data?" )
