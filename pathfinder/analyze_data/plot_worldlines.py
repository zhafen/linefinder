#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import galaxy_diver.plot_data.generic_plotter as generic_plotter
import galaxy_diver.plot_data.ahf as plot_ahf
import galaxy_diver.plot_data.plotting as gen_plot

import pathfinder.utils.presentation_constants as p_constants

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################


class WorldlinesPlotter( generic_plotter.GenericPlotter ):

    ########################################################################

    def plot_classification_bar(
        self,
        x_pos,
        classification_list = p_constants.CLASSIFICATIONS_A,
        classification_colors = p_constants.CLASSIFICATION_COLORS_B,
        ind = 0,
        ax = default,
        width = 0.5,
        add_label = False,
        alpha = p_constants.CLASSIFICATION_ALPHA,
        *args, **kwargs
    ):

        print( "Plotting bar at x_pos {}".format( x_pos ) )

        # Plot
        if ax is default:
            plt.figure( figsize=(11, 5), facecolor='white' )
            ax = plt.gca()

        classification_values = self.data_object.get_categories_selected_quantity_fraction(
            sl = (slice(None), ind),
            classification_list = classification_list,
            *args, **kwargs )

        bar_start = 0.
        for i, key in enumerate( classification_list ):

            if add_label:
                label = p_constants.CLASSIFICATION_LABELS[key]
            else:
                label = None

            value = classification_values[key]
            ax.bar(
                x_pos,
                value,
                width,
                bottom = bar_start,
                color = classification_colors[key],
                label = label,
                alpha = alpha,
            )

            bar_start += value

    ########################################################################

    def plot_classification_values(
        self,
        values = 'mass_fractions',
        ax = default,
        label = default,
        y_label = default,
        y_scale = default,
        color = default,
        pointsize = 3000,
        y_range = default,
    ):
        '''Plot overall values from a classification category.

        Args:
            ax (axis) : What axis to use. By default creates a figure and places the axis on it.
            label (str) : What label to use for the lines.
            color (str) : What color to use for the lines.
            pointsize (int) : What pointsize to use for the lines.
        '''

        if label is default:
            label = self.label
        if color is default:
            color = self.data_object.color

        print( "Plotting classification values for {}".format( label ) )

        classification_values = getattr( self.data_object, values )

        # Plot
        if ax is default:
            plt.figure( figsize=(11, 5), facecolor='white' )
            ax = plt.gca()

        objects = ( 'pristine', 'merger', 'intergalactic\ntransfer', 'wind' )
        x_pos = np.arange(len(objects))
        x_pos_dict = {
            'is_pristine': 0,
            'is_merger': 1,
            'is_mass_transfer': 2,
            'is_wind': 3,
        }

        for i, key in enumerate( classification_values.keys() ):
            if i != 0:
                label = None
            ax.scatter(
                x_pos_dict[key],
                classification_values[key],
                c=color,
                s=pointsize,
                marker='_',
                linewidths=5,
                vmin=0.5,
                vmax=1.5,
                label=label
            )

        plt.xticks( x_pos, objects, fontsize=22 )

        if y_label is default:
            y_label = values

        ax.set_ylabel( y_label, fontsize=22 )

        if y_range is not default:
            ax.set_ylim( y_range )

        if y_scale is not default:
            ax.set_yscale( y_scale )

    ########################################################################

    def plot_classified_time_dependent_data(
        self,
        ax = default,
        x_data = 'get_redshift',
        y_datas = 'get_categories_selected_quantity',
        x_range = [ 0., np.log10(8.) ], y_range = default,
        y_scale = 'log',
        x_label = default, y_label = default,
        classification_list = p_constants.CLASSIFICATIONS_A,
        classification_colors = p_constants.CLASSIFICATION_COLORS_B,
        *args, **kwargs
    ):
        '''Make a plot like the top panel of Fig. 3 in Angles-Alcazar+17

        Args:
            ax (axis object) :
                What axis to put the plot on. By default, create a new one on a separate figure.

            x_range, y_range (list-like) :
                [ x_min, x_max ] or [ y_min, y_max ] for the displayed range.

            x_label, y_label (str) :
                Labels for axis. By default, redshift and f(M_star), respectively.

            plot_dividing_line (bool) :
                Whether or not to plot a line at the edge between stacked regions.

            *args, **kwargs :
                Passed to the data retrieval method.
        '''

        if ax is default:
            plt.figure( figsize=(11, 5), facecolor='white' )
            ax = plt.gca()

        if x_data == 'get_redshift':
            x_data = np.log10( 1. + self.data_object.get_data( 'redshift' ) )

        if y_datas == 'get_categories_selected_quantity':
            y_datas = self.data_object.get_categories_selected_quantity(
                classification_list=classification_list,
                *args, **kwargs )

        for key in classification_list[::-1]:

            y_data = y_datas[key]

            ax.plot(
                x_data,
                y_data,
                linewidth = 3,
                color = classification_colors[key],
                label = p_constants.CLASSIFICATION_LABELS[key],
            )

        ax.set_xlim( x_range )

        if y_range is not default:
            ax.set_ylim( y_range )

        ax.set_yscale( y_scale )

        tick_redshifts = np.array( [ 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, ] )
        x_tick_values = np.log10( 1. + tick_redshifts )
        plt.xticks( x_tick_values, tick_redshifts )

        if y_label is default:
            y_label = r'$M_{\star} (M_{\odot})$'

        ax.set_xlabel( r'z', fontsize=22, )
        ax.set_ylabel( y_label, fontsize=22, )

        ax.annotate( s=self.label, xy=(0., 1.0225), xycoords='axes fraction', fontsize=22, )

        ax.legend( prop={'size': 14.5}, ncol=5, loc=(0., -0.28), fontsize=20 )

    ########################################################################

    def plot_stacked_time_dependent_data(
        self,
        ax = default,
        x_data = 'get_redshift',
        y_datas = 'get_categories_selected_quantity',
        x_range = [ 0., np.log10(8.)], y_range = [0., 1.],
        tick_redshifts = np.array( [ 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, ] ),
        x_label = default, y_label = default,
        plot_dividing_line = False,
        classification_list = p_constants.CLASSIFICATIONS_A,
        classification_colors = p_constants.CLASSIFICATION_COLORS_B,
        *args, **kwargs
    ):
        '''Make a plot like the bottom panel of Fig. 3 in Angles-Alcazar+17

        Args:
            ax (axis object) :
                What axis to put the plot on. By default, create a new one on a separate figure.

            x_range, y_range (list-like) :
                [ x_min, x_max ] or [ y_min, y_max ] for the displayed range.

            x_label, y_label (str) :
                Labels for axis. By default, redshift and f(M_star), respectively.

            plot_dividing_line (bool) :
                Whether or not to plot a line at the edge between stacked regions.
        '''

        if ax is default:
            plt.figure( figsize=(11, 5), facecolor='white' )
            ax = plt.gca()

        if x_data == 'get_redshift':
            x_data = np.log10( 1. + self.data_object.get_data( 'redshift' ) )

        if y_datas == 'get_categories_selected_quantity':
            y_datas = self.data_object.get_categories_selected_quantity_fraction(
                classification_list = classification_list,
                *args, **kwargs )

        y_prev = np.zeros( shape=y_datas.values()[0].shape )

        color_objects = []
        labels = []
        for key in classification_list[::-1]:

            y_next = y_prev + y_datas[key]

            ax.fill_between(
                x_data,
                y_prev,
                y_next,
                color = classification_colors[key],
                alpha = p_constants.CLASSIFICATION_ALPHA,
            )

            if plot_dividing_line:
                ax.plot(
                    x_data,
                    y_next,
                    linewidth = 1,
                    color = classification_colors[key],
                )

            y_prev = y_next

            # Make virtual artists to allow a legend to appear
            color_object = matplotlib.patches.Rectangle(
                (0, 0),
                1,
                1,
                fc = classification_colors[key],
                ec = classification_colors[key],
                alpha = p_constants.CLASSIFICATION_ALPHA,
            )
            color_objects.append( color_object )
            labels.append( p_constants.CLASSIFICATION_LABELS[key] )

        if x_range is not default:
            ax.set_xlim( x_range )

        if y_range is not default:
            ax.set_ylim( y_range )

        x_tick_values = np.log10( 1. + tick_redshifts )
        plt.xticks( x_tick_values, tick_redshifts )

        if y_label is default:
            y_label = r'$f(M_{\star})$'

        ax.set_xlabel( r'z', fontsize=22, )
        ax.set_ylabel( y_label, fontsize=22, )

        ax.annotate( s=self.label, xy=(0., 1.0225), xycoords='axes fraction', fontsize=22, )

        ax.legend( color_objects, labels, prop={'size': 14.5}, ncol=5, loc=(0., -0.28), fontsize=20 )

    ########################################################################

    def plot_stacked_radial_data(
        self,
        radial_bins,
        ax = default,
        x_range = default, y_range = [0., 1.],
        x_label = default, y_label = default,
        plot_dividing_line = False,
        classification_list = p_constants.CLASSIFICATIONS_A,
        classification_colors = p_constants.CLASSIFICATION_COLORS_B,
        *args, **kwargs
    ):
        '''

        Args:
            ax (axis object) :
                What axis to put the plot on. By default, create a new one on a separate figure.

            x_range, y_range (list-like) :
                [ x_min, x_max ] or [ y_min, y_max ] for the displayed range.

            x_label, y_label (str) :
                Labels for axis. By default, redshift and f(M_star), respectively.

            plot_dividing_line (bool) :
                Whether or not to plot a line at the edge between stacked regions.
        '''

        if ax is default:
            plt.figure( figsize=(11, 5), facecolor='white' )
            ax = plt.gca()

        y_datas = self.data_object.get_categories_selected_quantity_fraction(
            radial_bins = radial_bins,
            classification_list = classification_list,
            selected_quantity_method = 'get_selected_quantity_radial_bins',
            *args, **kwargs
        )

        y_prev = np.zeros( shape=y_datas.values()[0].shape )

        dr = radial_bins[1] - radial_bins[0]
        x_data = radial_bins[:-1] + dr / 2

        color_objects = []
        labels = []
        for key in classification_list[::-1]:

            y_next = y_prev + y_datas[key]

            ax.fill_between(
                x_data,
                y_prev,
                y_next,
                color = classification_colors[key],
                alpha = p_constants.CLASSIFICATION_ALPHA,
            )

            if plot_dividing_line:
                ax.plot(
                    x_data,
                    y_next,
                    linewidth = 1,
                    color = classification_colors[key],
                )

            y_prev = y_next

            # Make virtual artists to allow a legend to appear
            color_object = matplotlib.patches.Rectangle(
                (0, 0),
                1,
                1,
                fc = classification_colors[key],
                ec = classification_colors[key],
                alpha = p_constants.CLASSIFICATION_ALPHA,
            )
            color_objects.append( color_object )
            labels.append( p_constants.CLASSIFICATION_LABELS[key] )

        if x_range is default:
            ax.set_xlim( radial_bins.min(), radial_bins.max() )
        else:
            ax.set_xlim( x_range )

        if y_range is not default:
            ax.set_ylim( y_range )

        if y_label is default:
            y_label = r'Stacked Radial Data'
        if x_label is default:
            x_label = r'R'

        ax.set_xlabel( x_label, fontsize=22, )
        ax.set_ylabel( y_label, fontsize=22, )

        ax.annotate(
            s=self.label,
            xy=(0., 1.0225),
            xycoords='axes fraction',
            fontsize=22,
        )

        ax.legend(
            color_objects,
            labels,
            prop={'size': 14.5},
            ncol=5,
            loc=(0., -0.28),
            fontsize=20
        )

    ########################################################################

    def plot_dist_hist(
        self,
        data_key,
        ax,
        x_label = default,
        *args, **kwargs
    ):

        if x_label is default:
            if data_key == 'd_sat_scaled':
                if self.data_object.galids.parameters['length_scale'] == 'r_scale':
                    x_label = r'Distance to Nearest Other Galaxy ($r_{ \rm scale, sat }$)'
            elif data_key == 'd_sat_scaled_min':
                if self.data_object.galids.parameters['length_scale'] == 'r_scale':
                    x_label = r'Minimum Distance to Nearest Other Galaxy ($r_{ \rm scale, sat }$)'
        self.plot_hist( data_key, ax=ax, x_label=x_label, *args, **kwargs )

        # Add a bar indicating our radial cut
        trans = transforms.blended_transform_factory( ax.transData, ax.transAxes )
        r_cut = self.data_object.galids.parameters['galaxy_cut']
        ax.plot( [ r_cut, ] * 2, [0, 1], color='black', linewidth=3, linestyle='--', transform=trans )

    ########################################################################

    def plot_with_halos(
        self,
        plot_with_halos_method = 'histogram2d',
        slices = None,
        ax = default,
        out_dir = None,
        halo_color = '#337DB8',
        halo_linewidth = 3,
        halo_outline = False,
        *args, **kwargs
    ):

        if ax is default:
            fig = plt.figure( figsize=(10, 9), facecolor='white', )
            ax = plt.gca()

        used_plot_with_halos_method = getattr( self, plot_with_halos_method )
        used_plot_with_halos_method( ax = ax, slices=slices, *args, **kwargs )

        ahf_plotter = plot_ahf.AHFPlotter( self.data_object.ptracks.ahf_reader )
        snum = self.data_object.ptracks.ahf_reader.mtree_halos[0].index[slices]
        ahf_plotter.plot_halos_snapshot(
            snum,
            ax,
            color = halo_color,
            linewidth = halo_linewidth,
            outline = halo_outline,
            hubble_param = self.data_object.ptracks.data_attrs['hubble'],
            radius_fraction = self.data_object.galids.parameters['galaxy_cut'],
            length_scale = self.data_object.galids.parameters['length_scale'],
        )

        if out_dir is not None:
            save_file = '{}_{:03d}.png'.format( self.label, self.data_object.ptracks.snum[slices] )
            gen_plot.save_fig( out_dir, save_file, fig=fig, dpi=75 )

            plt.close()
