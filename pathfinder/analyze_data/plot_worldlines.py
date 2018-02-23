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
import matplotlib.collections as collections

import galaxy_diver.plot_data.generic_plotter as generic_plotter
import galaxy_diver.analyze_data.ahf as analyze_ahf
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
        label = default,
        show_label = True,
        label_kwargs = {
            'xy': (0., 1.0225),
            'xycoords': 'axes fraction',
            'fontsize': 22,
        },
        add_legend = True,
        legend_location = (0., -0.28),
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
        ax.xaxis.set_ticks( x_tick_values )
        ax.set_xticklabels( tick_redshifts )

        if y_label is default:
            y_label = r'$f(M_{\star})$'

        ax.set_xlabel( r'z', fontsize=22, )
        ax.set_ylabel( y_label, fontsize=22, )

        if label is default:
            label = self.label

        if show_label:
            ax.annotate( s=label, **label_kwargs )

        if add_legend:
            ax.legend(
                color_objects,
                labels,
                prop={'size': 14.5},
                ncol=5,
                loc=legend_location,
                fontsize=20
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
        *args, **kwargs
    ):
        '''Plot with halos overlayed on top.
        '''

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

    ########################################################################

    def plot_streamlines(
        self,
        x_key = 'Rx',
        y_key = 'Ry',
        x_data_kwargs = {},
        y_data_kwargs = {},
        classification = None,
        classification_ind = 0,
        start_ind = 0,
        end_ind = 100,
        sample_inds = default,
        sample_size = 10,
        ax = None,
        x_range = default,
        y_range = default,
        color = default,
        zorder = 100.,
        linewidth = 1.5,
        x_label = 'x position (pkpc)',
        y_label = 'y position (pkpc)',
        fontsize = 22,
        xkcd_mode = False,
        plot_halos = True,
        halo_radius_fraction = default,
        halo_length_scale = default,
        *args, **kwargs
    ):
        '''Plot streamlines. This code largely follows what Daniel did before.
        '''

        if xkcd_mode:
            plt.xkcd()

        if ax is None:
            plt.figure( figsize=(10, 8), facecolor='white' )
            ax = plt.gca()

        if sample_inds is default:
            if classification is None:

                inds_to_sample = range( self.data_object.n_particles )

            # Sample the data according to its classification at a
            # specified snapshot (index)
            else:
                inds = self.data_object.get_masked_data(
                    'ind_particle',
                    tile_data = True,
                    compress = False,
                    classification = classification,
                    *args, **kwargs
                )

                inds_to_sample = inds[:, classification_ind].compressed()

            # Get the indices to sample
            sample_inds = np.random.choice(
                inds_to_sample,
                sample_size,
            )

        print( "Displaying particles" )
        print( sample_inds )

        # Make the actual slice to pass to the data.
        x_slice = slice( start_ind, end_ind )
        sl = ( sample_inds, x_slice )

        # Account for possibly tiled x data (typically done if x data is
        # redshift or something similar).
        if 'tile_data' in x_data_kwargs:
            x_sl = x_slice
        else:
            x_sl = sl

        # Get the data out.
        x_data = self.data_object.get_masked_data(
            x_key,
            sl = x_sl,
            **x_data_kwargs
        )
        y_data = self.data_object.get_masked_data(
            y_key,
            sl = sl,
            **y_data_kwargs
        )

        # Format the data
        segments = []
        for i in range( len( sample_inds ) ):
            segment = np.array([ x_data[i], y_data[i] ]).transpose()

            segments.append( segment )

        # Plot!
        if color is default:
            color = p_constants.CLASSIFICATION_COLORS_B[classification]

        lc = collections.LineCollection(
            segments,
            color = color,
            linewidth = linewidth,
        )
        ax.add_collection( lc )

        lc.set_zorder( zorder )

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

        # Axis labels
        ax.set_xlabel( x_label, fontsize=fontsize )
        ax.set_ylabel( y_label, fontsize=fontsize )

    ########################################################################

    def plot_streamlines_vs_time(
        self,
        x_key = 'age_of_universe',
        y_key = 'Ry',
        classification_ind = 0,
        vert_line_at_classification_ind = True,
        horizontal_line_value = None,
        ax = None,
        x_label = 'Age of the Universe (Gyr)',
        plot_halos = True,
        halo_y_key = 'Yc',
        halo_plot_kwargs = {
            'n_halos': 100,
        },
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

        self.plot_streamlines(
            x_key = x_key,
            y_key = y_key,
            classification_ind = classification_ind,
            ax = ax,
            x_data_kwargs = x_data_kwargs,
            x_label = x_label,
            plot_halos = False,
            *args, **kwargs
        )

        # Plot a line at the ind at which classifications are determined
        if vert_line_at_classification_ind:

            x_value = self.data_object.get_masked_data(
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
                snums = w.get_masked_data( 'snum' ),
                subtract_mt_halo_id = w.main_halo_id,
                ax = ax,
                hubble_param = w.ptracks.data_attrs['hubble'],
                omega_matter = w.ptracks.data_attrs['omega_matter'],
                **halo_plot_kwargs
            )

    ########################################################################

    def plot_streamlines_old( self ):
        '''This should be deleted.'''

        def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
            '''
            Plot a colored line with coordinates x and y
            Optionally specify colors in the array z
            Optionally specify a colormap, a norm function and a line width
            '''

            # Default colors equally spaced on [0,1]:
            if z is None:
                z = np.linspace(0.0, 1.0, len(x))

            # Special case if a single number:
            if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
                z = np.array([z])

            z = np.asarray(z)

            segments = make_segments(x, y)
            lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

            ax = plt.gca()
            ax.add_collection(lc)

            return lc

        # --- INTERGALACTIC TRANSFER ---
        ind_gas_trans = np.where( (ptr['Ptype'][:, i] == 0) & (f['IsMassTransfer'][:] == 1) & (IsInOtherGal[:, i] == 1) & (IsAfterOtherGal[:, i - nstp] == 1) )[0]
        np.random.shuffle(ind_gas_trans)
        # for j in range(ind_gas_trans.size):
        for j in range( np.min([ind_gas_trans.size, ntest]) ):
            iacc = np.where( IsGasAccreted[ind_gas_trans[j], 0:i] == 1)[0]
            iaft = np.where( IsAfterOtherGal[ind_gas_trans[j], :] == 1)[0]
            if (iacc.size == 0) or (iaft.size == 0):
                continue
            iacc = iacc[-1]
            # iaft = iaft[-1]      ## this is the default!
            iaft = i
            if iacc > iaft:
                print 'problemssss...2'
            continue
            if (iaft - iacc < 2 * nsmooth):    # not enough snaps to smooth trajectory
                continue
            xtrs = daa.mysmooth(r[ind_gas_trans[j], iacc:iaft, 0], nsmooth, sfac=2.)
            ytrs = daa.mysmooth(r[ind_gas_trans[j], iacc:iaft, 1], nsmooth, sfac=2.)
            rtrs = np.sqrt(xtrs**2 + ytrs**2)
            ztrs = z[iacc:iaft]
            lwtrs = np.linspace(0.1, 1.5, xtrs.size)
            cltrs = np.linspace(0., 1, xtrs.size)
            colorline( xtrs, ytrs, z=cltrs, linewidth=lwtrs, cmap=cm.Greens, norm=cl.Normalize(vmin=-0.3, vmax=1.) )

    def plot_streamlines_classifications( self ):
        '''Plot multiple streamlines.
        '''

        # # --- read main galaxy info
        # if sname[0:3] == 'm13':
        #     grstr = 'gr1'
        # else:
        #     grstr = 'gr0'
        # skidname = simdir + 'skidgal_' + grstr + '.hdf5'
        # skidgal = h5py.File( skidname, 'r')

        # # --- read ptrack file
        # ptrack_name = simdir + 'ptrack_idlist_' + idtag + '.hdf5'
        # ptr = h5py.File(ptrack_name, 'r')

        # # --- read accmode file
        # GalDef = 2
        # WindVelMin = 15
        # WindVelMinVc = 2
        # TimeMin = 100.
        # TimeIntervalFac = 5.
        # neg = 5
        # accname = 'accmode_idlist_%s_g%dv%dvc%dt%dti%dneg%d.hdf5' % ( idtag, GalDef, WindVelMin, WindVelMinVc, TimeMin, TimeIntervalFac, neg )
        # f = h5py.File( simdir + accname, 'r')

        # nsnap = f['redshift'][:].size
        # snap_list = skidgal['snapnum'][0:nsnap]

        # # --- coordinates wrt galaxy center (physical kpc)
        # z = ptr['redshift'][0:nsnap]
        # hubble_factor = daa.hubble_z( z )  # h=header['hubble'], Omega0=header['Omega0'], OmegaLambda=header['OmegaLambda'] )
        # r = ptr['p'][:, 0:nsnap, :] - skidgal['p_phi'][0:nsnap, :]
        # R = np.sqrt((r * r).sum(axis=2))
        # v = ptr['v'][:, 0:nsnap, :] - skidgal['v_CM'][0:nsnap, :] + hubble_factor[:, np.newaxis] * r * UnitLength_in_cm / UnitVelocity_in_cm_per_s

        # # IsInGalID = ( ptr['GalID'][:, 0:nsnap] == skidgal['GalID'][0:nsnap] ).astype(int)
        # ind_rev = np.arange(nsnap - 2, -1, -1)
        # IsInGalID = f['IsInGalID'][:, :]
        # IsInOtherGal = ( (ptr['GalID'][:, 0:nsnap] > 0) & (IsInGalID == 0) ).astype(int)
        # IsAfterOtherGal = ( IsInOtherGal.cumsum(axis=1) == 0 ).astype(int)
        # IsGasAccreted = f['IsGasAccreted'][:, :]
        # IsEjected = f['IsEjected'][:, :]
        # CumNumEject = IsEjected[:, ind_rev].cumsum(axis=1)[:, ind_rev]      # cumulative number of EJECTION events
        # IsFirstEject = ( IsEjected & ( CumNumEject == 1 ) ).astype(int)

        # for i in range(nsnap):

        snapnum = snap_list[i]
        redshift = ptr['redshift'][i]

        # if round(redshift,3) not in [ 4., 3.5, 3., 2.5, 2., 1.5 ]:
        #  continue

        # --- FIGURE ---
        # figname = 'arr_' + g.snap_ext(snapnum) + '.pdf'
        figname = 'arr_s' + str(nstp) + '_' + g.snap_ext(snapnum) + '.png'
        fig = plt.figure( figsize=(4, 4) )
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        ax = fig.add_subplot(111)
        if rasterized:
            ax.set_rasterization_zorder(1)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(10)
        ax.set_aspect('equal')
        if round(redshift, 3) == 1.48:
            ax.text( 0.02, 0.94, 'z = ' + '%.2f' % (redshift), {'color': 'k', 'fontsize': 14}, transform=ax.transAxes )
        else:
            ax.text( 0.02, 0.94, 'z = ' + '%.1f' % (redshift), {'color': 'k', 'fontsize': 14}, transform=ax.transAxes )
        ax.set_xlim(xrange[0], xrange[1])
        ax.set_ylim(yrange[0], yrange[1])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.3)
        ax.xaxis.set_tick_params(width=1.3)
        ax.yaxis.set_tick_params(width=1.3)
        mew = 1

        # --- density grid
        P = g.readsnap( simdir, snapnum, 0, cosmological=1 )
        x = P['p'][:, 0] - skidgal['p_phi'][i, 0]
        y = P['p'][:, 1] - skidgal['p_phi'][i, 1]
        m = P['m'][:]
        hsml = P['h'][:]
        m_map = daa.make_2Dgrid( x, y, xrange, yrange=yrange, weight1=m, hsml=hsml, pixels=npix )

        ind_gas = np.where( ptr['Ptype'][:, i] == 0 )[0]
        """
        m = ptr['m'][:, i]
        rho = (4./3.) * np.pi * ptr['rho'][:, i]
        x = r[:, i, 0]  #ptr['p'][:, i, 0] - skidgal['p_phi'][i, 0]
        y = r[:, i, 1]
        hsml = (m[ind_gas]/rho[ind_gas])**(1./3) * ((MSUN/PROTONMASS)**(1./3)/CM_PER_KPC)
        m_map = daa.make_2Dgrid( x[ind_gas], y[ind_gas], xrange, yrange=yrange, weight1=m[ind_gas], hsml=hsml, pixels=npix )
        """

        m_map = daa.clip_2Dgrid( m_map )
        m_map /= m_map.max()
        # --- plot 2D grid
        m_map[m_map < 1e-4] = 1e-4
        vmin = np.log10(m_map.T).max() - 4.5
        vmax = np.log10(m_map.T).max() - 1
        im = ax.imshow( np.log10( m_map.T ), vmin=vmin, vmax=vmax, cmap=cm.Greys, interpolation='bicubic', origin='low', extent=[xrange[0], xrange[1], yrange[0], yrange[1]], zorder=0 )

        # --- PRISTINE ---
        m = ptr['m'][:, i]
        x = r[:, i, 0]
        y = r[:, i, 1]
        il = i - navg
        ir = i + navg
        if il < 0:
            il = 0
        vx = np.mean( v[:, il:ir, 0], axis=1 )
        vy = np.mean( v[:, il:ir, 1], axis=1 )
        vel = np.sqrt(vx**2 + vy**2)
        vx /= vel
        vy /= vel
        itr = (ptr['Ptype'][:, i] == 0) & (f['IsPristine'][:] == 1) & (f['IsWind'][:, i] == 0)
        mm_map, mvx_map, mvy_map = daa.make_2Dgrid( x[itr], y[itr], xrange, yrange=yrange, pixels=40, weight1=m[itr], weight2=m[itr] * vx[itr], weight3=m[itr] * vy[itr] )
        mm_map = daa.clip_2Dgrid( mm_map )
        vx_map = mvx_map / mm_map
        vy_map = mvy_map / mm_map
        Xgrid = np.linspace(xrange[0], xrange[1], mm_map.shape[0])
        Ygrid = np.linspace(yrange[0], yrange[1], mm_map.shape[1])
        # ax.streamplot( Xgrid, Ygrid, vx_map.T, vy_map.T, color='plum', density=[1., 1.], linewidth=0.8, arrowsize=1.5 )
        ax.streamplot( Xgrid, Ygrid, vx_map.T, vy_map.T, color='violet', density=[1., 1.], linewidth=1., arrowsize=2.3 )

        # --- WIND RECYCLING ---
        # ind_gas_wind = np.where( (ptr['Ptype'][:, i] == 0) & (f['IsPristine'][:] == 1) & (f['IsWind'][:, i] == 1) & (f['Neject'][:] > 1) )[0]
        ind_gas_wind = np.where( (ptr['Ptype'][:, i] == 0) & (IsInGalID[:, i] == 1) & (np.sum(IsEjected[:, i - nstp:i], axis=1) >= 1) )[0]
        np.random.shuffle(ind_gas_wind)
        # for j in range(ind_gas_wind.size):
        for j in range( np.min([ind_gas_wind.size, ntest]) ):
            iacc = np.where( IsGasAccreted[ind_gas_wind[j], 0:i] == 1)[0]
            if (iacc.size == 0):
                continue
            iacc = iacc[-1]
            iaft = i
            if iacc > iaft:
                print 'problemssss...1'
                continue
            if (iaft - iacc < 2 * nsmooth):    # not enough snaps to smooth trajectory
                continue
            xtrs = daa.mysmooth(r[ind_gas_wind[j], iacc:iaft, 0], nsmooth, sfac=2.)
            ytrs = daa.mysmooth(r[ind_gas_wind[j], iacc:iaft, 1], nsmooth, sfac=2.)
            rtrs = np.sqrt(xtrs**2 + ytrs**2)
            ztrs = z[iacc:iaft]
            lwtrs = np.linspace(0.1, 1.5, xtrs.size)
            cltrs = np.linspace(0., 1, xtrs.size)
            daa.colorline( xtrs, ytrs, z=cltrs, linewidth=lwtrs, cmap=cm.Blues, norm=cl.Normalize(vmin=-0.3, vmax=1.) )

        # --- INTERGALACTIC TRANSFER ---
        ind_gas_trans = np.where( (ptr['Ptype'][:, i] == 0) & (f['IsMassTransfer'][:] == 1) & (IsInOtherGal[:, i] == 1) & (IsAfterOtherGal[:, i - nstp] == 1) )[0]
        np.random.shuffle(ind_gas_trans)
        # for j in range(ind_gas_trans.size):
        for j in range( np.min([ind_gas_trans.size, ntest]) ):
            iacc = np.where( IsGasAccreted[ind_gas_trans[j], 0:i] == 1)[0]
            iaft = np.where( IsAfterOtherGal[ind_gas_trans[j], :] == 1)[0]
            if (iacc.size == 0) or (iaft.size == 0):
                continue
            iacc = iacc[-1]
            # iaft = iaft[-1]      ## this is the default!
            iaft = i
            if iacc > iaft:
                print 'problemssss...2'
            continue
            if (iaft - iacc < 2 * nsmooth):    # not enough snaps to smooth trajectory
                continue
            xtrs = daa.mysmooth(r[ind_gas_trans[j], iacc:iaft, 0], nsmooth, sfac=2.)
            ytrs = daa.mysmooth(r[ind_gas_trans[j], iacc:iaft, 1], nsmooth, sfac=2.)
            rtrs = np.sqrt(xtrs**2 + ytrs**2)
            ztrs = z[iacc:iaft]
            lwtrs = np.linspace(0.1, 1.5, xtrs.size)
            cltrs = np.linspace(0., 1, xtrs.size)
            daa.colorline( xtrs, ytrs, z=cltrs, linewidth=lwtrs, cmap=cm.Greens, norm=cl.Normalize(vmin=-0.3, vmax=1.) )
        # ax.scatter( r[ind_gas_trans, i, 0], r[ind_gas_trans, i, 1], s=15, c='lightgreen', edgecolors='green', alpha=0.2, zorder=0 )

        # --- plot star particle positions
        sz = 30
        """
        ind_star = np.where( ptr['Ptype'][:, i] == 4 )[0]
        np.random.shuffle(ind_star)
        if ind_star.size > ntest:
            ax.scatter( r[ind_star[0:ntest], i, 0], r[ind_star[0:ntest], i, 1], s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )
        else:
            ax.scatter( r[ind_star, i, 0], r[ind_star, i, 1], s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )
        """
        P = g.readsnap( simdir, snapnum, 4, cosmological=1 )
        x = P['p'][:, 0] - skidgal['p_phi'][i, 0]
        y = P['p'][:, 1] - skidgal['p_phi'][i, 1]
        nstar = x.size
        if nstar > ntestS:
            ind_star = np.arange(nstar)
            np.random.shuffle( ind_star )
            ax.scatter( x[ind_star[0:ntestS]], y[ind_star[0:ntestS]], s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )
        else:
            ax.scatter( x, y, s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )

        # --- plot SKID galaxies
        # for ng in xrange(Ngal-1):
        #   #if gal['npart'][ng] > 1000:
        #    xcen = gal['p_phi'][ng, 0] - cen_pos[0]
        #    ycen = gal['p_phi'][ng, 1] - cen_pos[1]
        #    #rcen = gal['Rout'][ng]
        #    rcen = 2. * gal['ReStar'][ng]
        #    ax.plot( xcen+rcen*xcirc, ycen+rcen*ycirc, '-', color='red', linewidth=lw)
        #    ax.plot( xcen, ycen, '+', color='red', ms=15, mew=mew )

        # --- plot Rvir for three most massive halos
        # halo = daa.read_AHF_halos(simdir + 'AHF', snapnum, readheader=False, h=P['hubble'] )
        halo = daa.read_AHF_halos(simdir, snapnum, readheader=False, h=h )
        for j in range(halo['Rvir'].size):
            if halo['M_star'][j] < 1e6:
                continue
        # for j in range(10):
        #   if halo['HostID'][j] >= 0:
        #     continue
            xcen = halo['pos'][j, 0] - skidgal['p_phi'][i, 0]
            ycen = halo['pos'][j, 1] - skidgal['p_phi'][i, 1]
            zcen = halo['pos'][j, 2] - skidgal['p_phi'][i, 2]
            rcen = halo['Rvir'][j]
            if np.abs(zcen) <= 2. * xrange[1]:
                ax.plot( xcen+rcen*xcirc, ycen+rcen*ycirc, '--', color='black', linewidth=0.5, zorder=10 )

        """
        #--- plot MAIN SKID galaxy
        xcen = 0
        ycen = 0
        rcen = 1. * skidgal['ReStar'][i]
        ax.plot( xcen+rcen*xcirc, ycen+rcen*ycirc, '-', color='red', linewidth=lw)
        """

        print 'snapnum = %d   redshift = %.4f %.4f    ngas = %d   nstar = %d' % ( snapnum, redshift, skidgal['redshift'][i], ind_gas.size, ind_star.size )
        sys.stdout.flush()

        fig.savefig(outdir + figname, rasterized=rasterized, dpi=150)
        plt.close()


        'Done!'
