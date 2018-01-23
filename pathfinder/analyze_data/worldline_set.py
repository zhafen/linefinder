#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import os
import string

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

import plot_worldlines
import worldlines as a_worldlines

import galaxy_diver.plot_data.plotting as gen_plot
import galaxy_diver.utils.utilities as utilities

########################################################################

# Sentinel object
default = object()

########################################################################
########################################################################

class WorldlineSet( utilities.SmartDict ):
    '''Container for multiple Worldlines classes. The nice thing about this class is you can use it like a
    Worldlines class, with the output being a dictionary of the different sets instead.
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

            worldlines_d[key] = { 'data_object' : a_worldlines.Worldlines( label=key, **kwargs ), 'label' : key }

        worldlines_plotters_d = utilities.SmartDict.from_class_and_args( plot_worldlines.WorldlinesPlotter, worldlines_d )

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
            filename_base = string.split( filename, '.' )[0]
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
        *args, **kwargs
    ):
        '''Iterate over each Worldlines class in the set, obtaining a
        specified quantity and then saving that to an .hdf5 file.
        '''

        quantity_method_used = getattr( self, quantity_method )

        quantities = quantity_method_used(
            *args, **kwargs
        )

    ########################################################################
    # Plotting Methods
    ########################################################################

    def plot_w_set_same_axis(
        self,
        plotting_method,
        *args, **kwargs
    ):

        fig = plt.figure( figsize=(11,5), facecolor='white' )
        ax = plt.gca()

        # The plot itself
        getattr( self, plotting_method )( ax=ax, *args, **kwargs )

        ax.legend(loc='upper center', prop={'size':14.5}, fontsize=20)

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

        fig = plt.figure( figsize=(11,5), facecolor='white' )
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
            kwargs[key]['x_pos'] = float( i ) - width/2.
            if i == 0:
                kwargs[key]['add_label'] = True

        self.plot_classification_bar.call_custom_kwargs( kwargs, default_kwargs )

        plt.xticks( range( len( data_order ) ), data_order, fontsize=22 )

        ax.set_xlim( [-0.5, len(self)-0.5] )
        ax.set_ylim( [0., 1.] )

        ax.set_ylabel( y_label, fontsize=22 )

        redshift = self[key].data_object.get_data( 'redshift' )[ind]
        title_string = r'$z=' + '{:.3f}'.format( redshift ) + '$'
        ax.set_title( title_string, fontsize=22, )

        if legend_args is default:
            ax.legend(prop={'size':14.5}, ncol=5, loc=(0.,-0.2), fontsize=20)
        else:
            ax.legend( **legend_args )

        if out_dir is not None:
            gen_plot.save_fig( out_dir, save_file=save_file, fig=fig )
