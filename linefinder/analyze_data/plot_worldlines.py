#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import imp
import git
import numpy as np
import os
import subprocess
import sys

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.collections as collections

import galaxy_dive.plot_data.generic_plotter as generic_plotter
import galaxy_dive.analyze_data.ahf as analyze_ahf
import galaxy_dive.plot_data.ahf as plot_ahf
import galaxy_dive.plot_data.plotting as gen_plot
import galaxy_dive.utils.astro as astro_utils

import linefinder.config as config
import linefinder.utils.presentation_constants as p_constants

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
        '''Make a bar plotting the classifications.
        '''

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

        ax.annotate(
            s = self.label,
            #xy = (0., 1.0225),
            xy = (0., 1.0),
            xycoords = 'axes fraction',
            va = 'bottom',
            fontsize = 22,
        )

        ax.legend( prop={'size': 14.5}, ncol=5, loc=(0., -0.28), fontsize=20 )

    ########################################################################

    def plot_stacked_time_dependent_data(
        self,
        ax = default,
        x_data = 'get_redshift',
        y_datas = 'get_categories_selected_quantity',
        x_range = [ 0., np.log10(8.) ], y_range = [0., 1.],
        tick_redshifts = np.array( [ 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, ] ),
        time_x_axis = False,
        twin_redshift = False,
        x_label = default, y_label = default,
        plot_dividing_line = False,
        classification_list = p_constants.CLASSIFICATIONS_A,
        classification_colors = p_constants.CLASSIFICATION_COLORS_B,
        label = default,
        show_label = True,
        label_kwargs = {
            'xy': (0., 1.0225),
            'xycoords': 'axes fraction',
            'fontsize': 22,
        },
        add_legend = True,
        hubble = None,
        omega_matter = None,
        legend_kwargs = {
            'loc' : 'center',
            'bbox_to_anchor' : (0.5, -0.3),
            'ncol' : 3,
        },
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

        if not time_x_axis:
            x_tick_values = np.log10( 1. + tick_redshifts )
            ax.xaxis.set_ticks( x_tick_values )
            ax.set_xticklabels( tick_redshifts )

        if twin_redshift:
            tick_times = astro_utils.age_of_universe(
                tick_redshifts,
                h = hubble,
                omega_matter = omega_matter,
                )

            # Make sure we aren't trying to plot ticks that would go out of bounds,
            # because that breaks things
            ax2_ticks = []
            ax2_tick_labels = []
            x_range = ax.get_xlim()
            for ax2_tick, ax2_tick_label in zip( tick_times, tick_redshifts ):
                if ( ax2_tick > x_range[0] ) and ( ax2_tick < x_range[1] ):
                    ax2_ticks.append( ax2_tick )
                    ax2_tick_labels.append( ax2_tick_label )

            # Add a second axis for plotting
            ax2 = ax.twiny()
            ax2.set_xlim( x_range )
            ax2.set_xticks( ax2_ticks )
            ax2.set_xticklabels( ax2_tick_labels )
            ax2.set_xlabel( r'z', fontsize=22, labelpad=10 )

        if y_label is default:
            y_label = r'$f(M_{\star})$'

        if not time_x_axis:
            ax.set_xlabel( r'z', fontsize=22, )
        else:
            ax.set_xlabel( r'Age of Universe (Gyr)', fontsize=22, )
        ax.set_ylabel( y_label, fontsize=22, )

        if label is default:
            label = self.label

        if show_label:
            ax.annotate( s=label, **label_kwargs )

        if add_legend:
            color_objects = color_objects[::-1]
            labels = labels[::-1]
            ax.legend(
                color_objects,
                labels,
                prop = {'size': 16},
                fontsize = 20,
                **legend_kwargs
            )

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
        '''Plot histogram of distances.
        '''

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
        radius_fractions = default,
        n_halos_plotted = 100,
        show_valid_halos = True,
        *args, **kwargs
    ):
        '''Plot with halos overlayed on top.
        '''

        if ax is default:
            fig = plt.figure( figsize=(10, 9), facecolor='white', )
            ax = plt.gca()

        used_plot_with_halos_method = getattr( self, plot_with_halos_method )
        used_plot_with_halos_method( ax = ax, slices=slices, *args, **kwargs )

        ahf_plotter = plot_ahf.HaloPlotter( self.data_object.ptracks.halo_data )
        snum = self.data_object.ptracks.ahf_reader.mtree_halos[0].index[slices]

        if radius_fractions is default:
            radius_fractions = [ self.data_object.galids.parameters['galaxy_cut'], ]

        for radius_fraction in radius_fractions:
            ahf_plotter.plot_halos_snapshot(
                snum,
                ax,
                color = halo_color,
                linewidth = halo_linewidth,
                outline = halo_outline,
                hubble_param = self.data_object.ptracks.data_attrs['hubble'],
                radius_fraction = radius_fraction,
                length_scale = self.data_object.galids.parameters['length_scale'],
                n_halos_plotted = n_halos_plotted,
                show_valid_halos = show_valid_halos,
            )

        if out_dir is not None:
            save_file = '{}_{:03d}.png'.format( self.label, self.data_object.ptracks.snum[slices] )
            gen_plot.save_fig( out_dir, save_file, fig=fig, resolution=75 )

            plt.close()

    ########################################################################

    def plot_streamlines(
        self,
        x_key = 'Rx',
        y_key = 'Ry',
        x_data_kwargs = {},
        y_data_kwargs = {},
        classification = None,
        classification_ind = 0,
        sample_selected_interval = True,
        t_show_min = 0.5,
        t_show_max = 1.0,
        start_ind = 0,
        end_ind = 'time_based',
        t_end = default,
        sample_inds = default,
        sample_size = 10,
        convert_x_to_comoving = False,
        convert_y_to_comoving = False,
        ax = None,
        x_range = default,
        y_range = default,
        x_scale = 'linear',
        y_scale = 'linear',
        color = default,
        zorder = 100.,
        linewidth = 1.5,
        fade_streamlines = True,
        x_label = 'x position (pkpc)',
        y_label = 'y position (pkpc)',
        plot_xlabel = True,
        plot_ylabel = True,
        fontsize = 22,
        xkcd_mode = False,
        plot_halos = True,
        halo_radius_fraction = default,
        halo_length_scale = default,
        verbose = False,
        return_y_max = False,
        *args, **kwargs
    ):
        '''Plot streamlines. This code largely follows what Daniel did before,
        at least in ideas.
        '''

        if xkcd_mode:
            plt.xkcd()

        if ax is None:
            plt.figure( figsize=(10, 8), facecolor='white' )
            ax = plt.gca()

        if sample_selected_interval:

            time_key = 'time_as_{}'.format( classification[3:] )
            self.data_object.data_masker.mask_data(
                time_key, t_show_min, t_show_max )

        # Decide when to stop plotting the streamlines
        if end_ind == 'time_based':

            # Figure out what time interval overwhich to plot
            if t_end is default:
                t_end = t_show_max

            # Loop through until we get the right time.
            time = self.data_object.get_data('time')
            end_ind = start_ind
            time_end = time[end_ind]
            while time[start_ind] - time_end <= t_end:
                time_end = time[end_ind]
                end_ind += 1

        if sample_inds is default:
            if classification is None:

                inds_to_sample = range( self.data_object.n_particles )

            # Sample the data according to its classification at a
            # specified snapshot (index)
            else:
                inds = self.data_object.get_selected_data(
                    'ind_particle',
                    tile_data = True,
                    compress = False,
                    classification = classification,
                    *args, **kwargs
                )

                self.data_object.data_masker.clear_masks()

                inds_to_sample = inds[:, classification_ind].compressed()

            # Get the indices to sample
            sample_inds = np.random.choice(
                inds_to_sample,
                sample_size,
            )

        if verbose:
            print( "Displaying particles" )
            print( sample_inds )

        # Make the actual slice to pass to the data.
        onedim_slice = slice( start_ind, end_ind )
        sl = ( sample_inds, onedim_slice )

        # Get the data out.
        x_data = self.data_object.get_selected_data(
            x_key,
            sl = sl,
            **x_data_kwargs
        )
        y_data = self.data_object.get_selected_data(
            y_key,
            sl = sl,
            **y_data_kwargs
        )

        # Convert to comoving
        if convert_x_to_comoving or convert_y_to_comoving:
            a = ( 1. + self.data_object.redshift.values[onedim_slice] )**-1.

        if convert_x_to_comoving:
            x_data /= a
        if convert_y_to_comoving:
            y_data /= a

        # Data for the streamlines color
        z_data = np.linspace(0., 1., x_data.shape[1] )

        # Make it so that we spend more time at the specified color.
        # (This goes from 0 to 1, and the more time at 0, the less time
        # the color is white.)
        z_data = z_data**2.

        # Plot!
        if color is default:
            color = p_constants.CLASSIFICATION_COLORS_B[classification]

        # Format the data
        for i in range( len( sample_inds ) ):

            xs = x_data[i]
            ys = y_data[i]

            if fade_streamlines:
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = collections.LineCollection(
                    segments,
                    cmap = gen_plot.custom_sequential_colormap( color ),
                    # color = color,
                    norm = plt.Normalize(0, 1),
                    linewidths = np.linspace( linewidth, 0, x_data.shape[1] ),
                    array = z_data,
                )
                ax.add_collection( lc )

                lc.set_zorder( zorder )
            else:
                ax.plot(
                    xs,
                    ys,
                    linewidth = linewidth,
                    zorder = zorder,
                    color = color
                )

        # Plot halos
        if plot_halos:

            w = self.data_object

            if halo_radius_fraction is default:
                halo_radius_fraction = w.galids.parameters['galaxy_cut']
            if halo_length_scale is default:
                halo_length_scale = w.galids.parameters['length_scale']

            ahf_data = analyze_ahf.HaloData(
                w.ahf_data_dir,
                tag = w.ahf_tag,
                index = w.ahf_index,
            )
            ahf_plotter = plot_ahf.HaloPlotter( ahf_data )

            ahf_plotter.plot_halos_snapshot(
                snum = w.ahf_index - classification_ind,
                ax = ax,
                hubble_param = w.ptracks.data_attrs['hubble'],
                radius_fraction = halo_radius_fraction,
                length_scale = halo_length_scale,
                minimum_criteria = w.galids.parameters['minimum_criteria'],
                minimum_value = w.galids.parameters['minimum_value'],
            )

        # Set the range
        if x_range is default:
            x_range = [ x_data.min(), x_data.max() ]
        if y_range is default:
            y_range = [ y_data.min(), y_data.max() ]
        ax.set_xlim( x_range )
        ax.set_ylim( y_range )

        # Set the scale
        ax.set_xscale( x_scale )
        ax.set_yscale( y_scale )

        # Axis labels
        if plot_xlabel:
            ax.set_xlabel( x_label, fontsize=fontsize )
        if plot_ylabel:
            ax.set_ylabel( y_label, fontsize=fontsize )

        if return_y_max:
            return np.nanmax( y_data )

    ########################################################################

    def plot_streamlines_vs_time(
        self,
        x_key = 'time',
        y_key = 'R',
        classification_ind = 0,
        vert_line_at_classification_ind = True,
        sample_selected_interval = False,
        horizontal_line_value = None,
        ax = None,
        fade_streamlines = False,
        x_label = 'Age of the Universe (Gyr)',
        y_label = 'Radial Distance (pkpc)',
        plot_halos = False,
        halo_y_key = 'Yc',
        halo_plot_kwargs = {
            'n_halos': 100,
        },
        plot_CGM_region = False,
        return_y_max = False,
        *args, **kwargs
    ):

        if ax is None:
            plt.figure( figsize=(10, 8), facecolor='white' )
            ax = plt.gca()

        # Setup data kwargs. These will always have to be tiled
        x_data_kwargs = {
            'tile_data': True,
            'tile_dim': 'match_particles',
        }

        y_max = self.plot_streamlines(
            x_key = x_key,
            y_key = y_key,
            classification_ind = classification_ind,
            sample_selected_interval = sample_selected_interval,
            ax = ax,
            x_data_kwargs = x_data_kwargs,
            x_label = x_label,
            y_label = y_label,
            plot_halos = False,
            fade_streamlines = fade_streamlines,
            return_y_max = return_y_max,
            *args, **kwargs
        )

        # Plot a line at the ind at which classifications are determined
        if vert_line_at_classification_ind:

            x_value = self.data_object.get_selected_data(
                x_key,
                sl = classification_ind,
            )

            trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes )

            ax.plot(
                [ x_value, ] * 2,
                [0, 1],
                color='black',
                linewidth=3,
                linestyle='--',
                transform=trans
            )
        if horizontal_line_value is not None:

            trans = transforms.blended_transform_factory(
                ax.transAxes, ax.transData )

            ax.plot(
                [0, 1],
                [ horizontal_line_value, ] * 2,
                color='black',
                linewidth=3,
                linestyle='--',
                transform=trans
            )

        # Plot halos
        if plot_halos:

            w = self.data_object

            ahf_data = analyze_ahf.HaloData(
                w.ahf_data_dir,
                tag = w.ahf_tag,
                index = w.ahf_index,
            )
            ahf_plotter = plot_ahf.HaloPlotter( ahf_data )

            ahf_plotter.plot_halo_time(
                halo_y_key,
                snums = w.get_selected_data( 'snum' ),
                subtract_mt_halo_id = w.main_halo_id,
                ax = ax,
                hubble_param = w.ptracks.data_attrs['hubble'],
                omega_matter = w.ptracks.data_attrs['omega_matter'],
                **halo_plot_kwargs
            )

        # Plot a shaded region showing the CGM
        if plot_CGM_region:

            # Plot multiple directions only if not using radial distance.
            if y_key == 'R':
                y_dirs = [ 1., ]
            else:
                y_dirs = [ 1., -1. ]

            for y_dir in y_dirs:
                ax.fill_between(
                    self.data_object.get_data( 'time' ),
                    y_dir * self.data_object.inner_CGM_boundary,
                    y_dir * self.data_object.outer_CGM_boundary,
                    color = 'k',
                    alpha = 0.2,
                )

        if return_y_max:
            return y_max

    ########################################################################

    def export_to_firefly(
        self,
        firefly_dir,
        install_firefly = False,
        firefly_source = 'https://github.com/ageller/Firefly.git',
        write_startup = 'append',
        pathlines = True,
        n_pathlines = 100,
        snum = 600,
        classifications = [ None, ],
        classification_ui_labels = [ 'All' ],
        tracked_properties = [
            'T',
            'Z',
            'Den',
            'is_in_main_gal',
            'is_in_other_gal',
            'PType',
        ],
        log_properties = [ 'T', 'Z', 'Den', ],
        tracked_filter_flags = [ True, ] * 6,
        tracked_colormap_flags = [ True, True, True, False, False, False, ],
    ):

        if install_firefly:

            print( "Cloning Firefly..." )

            # Make and switch to the directory containing firefly
            containing_dir = os.path.dirname( firefly_dir )

            if not os.path.isdir( containing_dir ):
                os.makedirs( containing_dir )
            os.chdir( containing_dir )

            # Clone
            git.Repo.clone_from( firefly_source, firefly_dir )

            # Validate install
            if os.path.isfile( os.path.join( firefly_dir, 'index.html' ) ):
                print(
                    "Successfully cloned Firefly at {}".format( firefly_dir )
                )
            else:
                raise Exception( "Failed to install Firefly." )

        # Import the Firefly data parser
        sys.path.append(
            os.path.join( firefly_dir, 'data' )
        )
        import dataParser

        # Make the JSON dir exist if it doesn't
        json_dir = os.path.join(
            firefly_dir,
            'data',
            self.data_object.tag,
        )
        if not os.path.isdir( json_dir ):
            os.makedirs( json_dir )

        # Setup a reader
        firefly_reader = dataParser.Reader(
            JSONdir = json_dir,
            write_startup = write_startup,
            clean_JSONdir = True,
        )

        def get_data(
            data_key,
            classification,
            seed = None,
            *args, **kwargs
        ):
            '''Function for inserting linefinder data into Firefly.
            '''

            if pathlines:

                return self.data_object.get_selected_data_over_time(
                    data_key,
                    snum = snum,
                    classification = classification,
                    n_samples = n_pathlines,
                    seed = seed,
                    *args, **kwargs
                ).flatten()
                
            else:
               
                return self.data_object.get_selected_data(
                    data_key,
                    sl = (slice(None),ind),
                    classification = classification,
                    *args, **kwargs
                )

        for i, classification in enumerate( classifications ):

            # We choose a random seed for each classification.
            # When doing time data this is important for making sure we
            # select the same data
            seed = np.random.randint( 1e7 )

            # Grabbing the position and velocity
            coords = []
            vels = []
            for pos_key, vel_key in zip(
                [ 'Rx', 'Ry', 'Rz' ],
                [ 'Vx', 'Vy', 'Vz' ],
            ):

                # Get position data
                ri = get_data(
                    pos_key,
                    classification = classification,
                    seed = seed,
                )
                coords.append( ri )

                # Get velocity data
                vi = get_data(
                    vel_key,
                    classification = classification,
                    seed = seed,
                )
                vels.append( vi )
                    
            # Make coords and velocities into a numpy array
            coords = np.array( coords ).transpose()
            vels = np.array( vels ).transpose()

            tracked_arrs = []
            tracked_labels = []
            for tracked_key in tracked_properties:

                # Add the Array
                tracked_arr = get_data(
                    tracked_key,
                    classification = classification,
                    seed = seed,
                )
                tracked_arrs.append( tracked_arr )

                # Add the label
                # TODO: When I implement the log part then this needs
                # to be updated to account for that.
                tracked_labels.append(
                    tracked_key
                )

            # Create a particle group and add to the firefly reader
            particle_group = dataParser.ParticleGroup(
                UIname = classification_ui_labels[i],
                coordinates = coords,
                tracked_arrays = tracked_arrs,
                tracked_names = tracked_labels,
                tracked_filter_flags = tracked_filter_flags,
                tracked_colormap_flags = tracked_colormap_flags,
            )
            firefly_reader.addParticleGroup( particle_group )

        # Finish up and write data
        firefly_reader.dumpToJSON()
