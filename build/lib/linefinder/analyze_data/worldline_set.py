#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import numpy as np
import os
import string
import verdict

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

from . import plot_worldlines
from . import worldlines as a_worldlines

import galaxy_dive.plot_data.plotting as gen_plot
import galaxy_dive.utils.hdf5_wrapper as hdf5_wrapper

########################################################################

# Sentinel object
default = object()

########################################################################
########################################################################


class WorldlineSet( verdict.Dict ):
    '''Container for multiple Worldlines classes. The nice thing about this
    class is you can use it like a Worldlines class, with the output being a
    dictionary of the different sets instead.
    '''

    def __init__( self, defaults, variations ):
        '''
        Args:
            defaults (dict) :
                Set of default arguments for loading worldline data.

            variations (dict of dicts) :
                Labels and differences in arguments to be passed to Worldlines
        '''

        # Load the worldline sets
        worldlines_d = {}
        for key in variations.keys():

            kwargs = dict( defaults )
            for var_key in variations[key].keys():
                kwargs[var_key] = variations[key][var_key]

            worldlines_d[key] = {
                'data_object': a_worldlines.Worldlines( label=key, **kwargs ),
                'label': key
            }

        worldlines_plotters_d = verdict.Dict.from_class_and_args(
            plot_worldlines.WorldlinesPlotter, worldlines_d )

        super( WorldlineSet, self ).__init__( worldlines_plotters_d )

    ########################################################################

    @classmethod
    def from_tag_expansion( cls, defaults, tag_expansion ):
        '''Create a worldline set using a bash-style expansion for the
        variations.

        Args:
            defaults (dict) :
                Set of default arguments for loading worldline data.

            tag_expansion (str) :
                String that can be expanded through wildcards to find
                different worldline data. For example, 'analyze_*' would look
                for all data files with tags starting with 'analyze' and ending
                with something else;

        Returns:
            worldline_set (WorldlineSet instance)
        '''

        # Get the paths after everything's expanded
        filepath_unexpanded = os.path.join(
            defaults['data_dir'],
            'ptracks_{}.hdf5'.format( tag_expansion ),
        )
        filepaths = glob.glob( filepath_unexpanded )

        # Now get the variation dictionary out
        variations = {}
        for filepath in filepaths:

            # Get the tag
            filename = os.path.split( filepath )[1]
            filename_base = filename.split( '.' )[0]
            tag = filename_base[8:]

            # Add the variation to the dictionary
            variations[tag] = {}
            variations[tag]['tag'] = tag

        return cls( defaults, variations )

    ########################################################################
    # Data Analysis Methods
    ########################################################################

    def store_quantity(
        self,
        output_filepath,
        quantity_method = 'get_categories_selected_quantity_fraction',
        variations = None,
        verbose = True,
        *args, **kwargs
    ):
        '''Iterate over each Worldlines class in the set, obtaining a
        specified quantity and then saving that to an .hdf5 file.
        Note that verdict.Dict's to_hdf5 method is simpler and more functional
        for many cases.

        Args:
            output_filepath (str) :
                Full path to store the output at.

            quantity_method (str) :
                What method to use for getting the quantity out.

            variations (dict) :
                If provided, we get different quantities from each Worldlines
                instance by passing different args to quantity_method according
                to variations. kwargs is taken as the default arguments when
                none are specified.

            *args, **kwargs :
                Arguments to be passed to quantity_method.
        '''

        quantity_method_used = getattr( self, quantity_method )

        if variations is None:
            quantities = quantity_method_used(
                *args, **kwargs
            )
        else:
            assert len(args) == 0, \
                "variations arg is not compatible with calling" + \
                " quantity_method with *args."

            quantities = quantity_method_used.call_custom_kwargs(
                variations, kwargs, verbose=verbose )

        # # Setup data to store
        # data_to_store = {}
        # for data_category in list( quantities.values() )[0].keys():
        #     data_to_store[data_category] = []

        # # Format the data
        # labels = []
        # for label, item in quantities.items():

        #     # Store the data
        #     for data_category, inner_item in quantities[label].items():
        #         data_to_store[data_category].append( inner_item )

        #     # Store what tag was used.
        #     labels.append( label )
        # data_to_store['label'] = np.array( labels )

        # Store data
        data_to_store = {}
        labels_stored = False
        for key, item in quantities.transpose().items():
            data_to_store[key] = item.array()

            # Store labels
            if not labels_stored:
                data_to_store['label'] = item.keys_array()

        h5_wrapper = hdf5_wrapper.HDF5Wrapper( output_filepath )
        h5_wrapper.save_data( data_to_store, index_key='label' )

    ########################################################################

    def store_redshift_dependent_quantity(
        self,
        output_filepath,
        max_snum,
        choose_snum_by = 'pulling_from_ids',
        *args, **kwargs
    ):
        '''Store a redshift dependent quantity. In particular, this method
        assumes that the differences between the Worldlines is a redshift
        difference, and that the label for each Worldlines class contains
        the relevant snapshot number. It then parses the label for that
        snapshot number, and passes it to store_dependent_quantity as a key.

        Args:
            output_filepath (str) :
                Where to save the output data.

            max_snum (int) :
                Maximum snapshot number the snapshots can go to. Used for parsing
                the label as well as specifying the ind.

            choose_snum_by (str) :
                How to to vary the snapshot used when the arguments are passed
                to the Worldlines.

            *args, **kwargs :
                Arguments passed to self.store_dependent_quantity()
        '''

        # Rename keys
        variations = {}
        labels = []
        snums = []
        for key, item in self.items():

            if choose_snum_by == 'parsing_tag':
                # Parse the tag for the snapshot number
                snum_str_ind = key.find( 'snum' )
                snum_ind_start = snum_str_ind + 4
                snum_ind_end = snum_str_ind + 4 + len( str( max_snum ) )
                snum_str = key[snum_ind_start:snum_ind_end]
                # Account for when snum isn't 3 digits long
                if snum_str[2] == '_':
                    snum_str = snum_str[:2]
                snum = int( snum_str )

            elif choose_snum_by == 'pulling_from_ids':
                snum = item.data_object.ids.parameters['snum_end']

                # Make sure we have IDs for which is is valid
                assert snum == item.data_object.ids.parameters['snum_start'], \
                    "Automated snum selection method not valid for given data."

            ind = max_snum - snum

            # Setup variations dict
            variations[key] = {
                'sl': (slice(None), ind),
            }

            # Store the label and snum.
            labels.append( key )
            snums.append( snum )

        self.store_quantity(
            output_filepath,
            variations = variations,
            *args, **kwargs
        )

        # Now store the snapshots
        h5_wrapper = hdf5_wrapper.HDF5Wrapper( output_filepath )
        data_to_store = {
            'snum': snums,
            'label': labels,
        }
        h5_wrapper.insert_data(
            new_data = data_to_store,
            index_key = 'label',
            insert_columns = True,
        )

    ########################################################################
    # Plotting Methods
    ########################################################################

    def plot_w_set_same_axis(
        self,
        plotting_method,
        *args, **kwargs
    ):

        plt.figure( figsize=(11, 5), facecolor='white' )
        ax = plt.gca()

        # The plot itself
        getattr( self, plotting_method )( ax=ax, *args, **kwargs )

        ax.legend(loc='upper center', prop={'size': 14.5}, fontsize=20)

    ########################################################################

    def plot_classification_bar_same_axis(
        self,
        kwargs = default,
        ind = 0,
        width = 0.5,
        data_order = default,
        legend_args = default,
        y_label = 'Classification Fraction',
        out_dir = None,
        save_file = 'bar_map.pdf',
        **default_kwargs
    ):

        fig = plt.figure( figsize=(11, 5), facecolor='white' )
        ax = plt.gca()

        default_kwargs['width'] = width
        default_kwargs['ind'] = ind

        if data_order is default:
            data_order = self.keys()

        if kwargs is default:
            kwargs = {}
            for key in data_order:
                kwargs[key] = {}

        # Pass automated arguments to plot_classification_bar
        for i, key in enumerate( data_order ):
            kwargs[key]['ax'] = ax
            kwargs[key]['x_pos'] = float( i ) - width / 2.
            if i == 0:
                kwargs[key]['add_label'] = True

        self.plot_classification_bar.call_custom_kwargs(
            kwargs, default_kwargs )

        plt.xticks( range( len( data_order ) ), data_order, fontsize=22 )

        ax.set_xlim( [-0.5, len(self) - 0.5] )
        ax.set_ylim( [0., 1.] )

        ax.set_ylabel( y_label, fontsize=22 )

        redshift = self[key].data_object.get_data( 'redshift' )[ind]
        title_string = r'$z=' + '{:.3f}'.format( redshift ) + '$'
        ax.set_title( title_string, fontsize=22, )

        if legend_args is default:
            ax.legend(prop={'size': 14.5}, ncol=5, loc=(0., -0.2), fontsize=20)
        else:
            ax.legend( **legend_args )

        if out_dir is not None:
            gen_plot.save_fig( out_dir, save_file=save_file, fig=fig )
