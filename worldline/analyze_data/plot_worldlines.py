#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import os

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

import galaxy_diver.plot_data.generic_plotter as generic_plotter
import galaxy_diver.plot_data.ahf as plot_ahf
import galaxy_diver.plot_data.plotting as gen_plot
import galaxy_diver.plot_data.pu_colormaps as pu_cm
import galaxy_diver.utils.mp_utils as mp_utils
import galaxy_diver.utils.utilities as utilities

import analyze_worldlines

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class WorldlinesPlotter( generic_plotter.GenericPlotter ):

  def plot_classification_values( self,
    ax = default,
    label = default,
    color = default,
    pointsize = 3000,
    ):
    '''Plot overall values from a classification category.

    Args:
      ax (axis) : What axis to use. By default creates a figure and places the axis on it.
    '''

    if label is default:
      label = self.label
    if color is default:
      color = self.color

    print( "Plotting classification values for {}".format( label ) )

    classification_values = self.mass_fractions

    # Plot
    if ax is default:
      fig = plt.figure( figsize=(11,5), facecolor='white' )
      ax = plt.gca()

    objects = ( 'fresh\naccretion', 'merger', 'intergalactic\ntransfer', 'wind' )
    x_pos = np.arange(len(objects))
    x_pos_dict = {
      'is_pristine' : 0,
      'is_merger' : 1,
      'is_mass_transfer' : 2,
      'is_wind' : 3,
    }

    for i, key in enumerate( classification_values.keys() ):
      if i != 0:
        label = None
      ax.scatter( x_pos_dict[key], classification_values[key], c=color, s=pointsize,
                    marker='_', linewidths=5, vmin=0.5, vmax=1.5, label=label )

    plt.xticks( x_pos, objects, fontsize=22 )

    ax.set_ylabel( r'$\frac{ m_{\rm{class}} }{ m_{\rm{ total }} }$', fontsize=35, rotation=0, labelpad=35 )
    ax.set_ylim( [ 0., 1. ])

  ########################################################################

  def plot_hist_2d( self,
    x_key, y_key,
    slices,
    ax = default,
    x_range = default, y_range = default,
    n_bins = 128,
    vmin = None, vmax = None,
    plot_halos = False,
    add_colorbar = True,
    colorbar_args = default,
    x_label = default, y_label = default,
    add_x_label = True, add_y_label = True,
    plot_label = default,
    outline_plot_label = False,
    label_galaxy_cut = True,
    label_redshift = True,
    label_fontsize = 24,
    tick_param_args = default,
    out_dir = None,
    *args, **kwargs ):
    '''Make a 2D histogram of the data. Extra arguments are passed to get_masked_data.
    Args:
      x_key, y_key (str) : Data keys to plot.
      slices (int or tuple of slices) : How to slices the data.
      ax (axis) : What axis to use. By default creates a figure and places the axis on it.
      x_range, y_range ( (float, float) ) : Histogram edges. If default, all data is enclosed. If list, set manually.
        If float, is +- x_range*length scale at that snapshot.
      n_bins (int) : Number of bins in the histogram.
      vmin, vmax (float) : Limits for the colorbar.
      plot_halos (bool) : Whether or not to plot merger tree halos on top of the histogram.
        Only makes sense for when dealing with positions.
      add_colorbar (bool) : If True, add a colorbar to colorbar_args
      colorbar_args (axis) : What axis to add the colorbar to. By default, is ax.
      x_label, ylabel (str) : Axes labels.
      add_x_label, add_y_label (bool) : Include axes labels?
      plot_label (str or dict) : What to label the plot with. By default, uses self.label.
        Can also pass a dict of full args.
      outline_plot_label (bool) : If True, add an outline around the plot label.
      label_galaxy_cut (bool) : If true, add a label that indicates how the galaxy was defined.
      label_redshift (bool) : If True, add a label indicating the redshift.
      label_fontsize (int) : Fontsize for the labels.
      tick_param_args (args) : Arguments to pass to ax.tick_params. By default, don't change inherent defaults.
      out_dir (str) : If given, where to save the file.
    '''

    if isinstance( slices, int ):
      sl = ( slice(None), slices )

    # Get data
    x_data = self.get_masked_data( x_key, sl=sl, *args, **kwargs )
    y_data = self.get_masked_data( y_key, sl=sl, *args, **kwargs )

    if x_range is default:
      x_range = [ x_data.min(), x_data.max() ]
    elif isinstance( x_range, float ):
      x_range = np.array( [ -x_range, x_range ])*self.ptracks.length_scale.iloc[slices]
    if y_range is default:
      y_range = [ y_data.min(), y_data.max() ]
    elif isinstance( y_range, float ):
      y_range = np.array( [ -y_range, y_range ])*self.ptracks.length_scale.iloc[slices]

    x_edges = np.linspace( x_range[0], x_range[1], n_bins )
    y_edges = np.linspace( y_range[0], y_range[1], n_bins )

    # Make the histogram
    hist2d, x_edges, y_edges = np.histogram2d( x_data, y_data, [x_edges, y_edges] )

    # Plot
    if ax is default:
      fig = plt.figure( figsize=(10,9), facecolor='white' )
      ax = plt.gca()

    im = ax.imshow( np.log10( hist2d ).transpose(), cmap=pu_cm.magma, interpolation='nearest',\
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], \
                    origin='low', aspect='auto', vmin=vmin, vmax=vmax, )

    # Add a colorbar
    if add_colorbar:
      if colorbar_args is default:
        colorbar_args = ax
        cbar = gen_plot.add_colorbar( colorbar_args, im, method='ax' )
      else:
        colorbar_args['color_object'] = im
        cbar = gen_plot.add_colorbar( **colorbar_args )
      cbar.ax.tick_params( labelsize=20 )

    # Halo Plot
    if plot_halos:
      ahf_plotter = plot_ahf.AHFPlotter( self.ptracks.ahf_reader )
      snum = self.ptracks.ahf_reader.mtree_halos[0].index[slices]
      ahf_plotter.plot_halos_snapshot( snum, ax, hubble_param=self.ptracks.data_attrs['hubble'],
        radius_fraction=self.galids.parameters['galaxy_cut'] )
      assert self.galids.parameters['length_scale'] == 'r_scale'

    # Plot label
    if plot_label is default:
      plt_label = ax.annotate( s=self.label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
    elif isinstance( plot_label, str ):
      plt_label = ax.annotate( s=plot_label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
    elif isinstance( plot_label, dict ):
      plt_label = ax.annotate( **plot_label )
    else:
      raise Exception( 'Unrecognized plot_label arguments, {}'.format( plot_label ) )
    if outline_plot_label:
      plt_label.set_path_effects([ path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal() ])

    # Upper right label (info label)
    info_label = ''
    if label_galaxy_cut:
      info_label = r'$r_{ \rm cut } = ' + '{:.3g}'.format( self.galids.parameters['galaxy_cut'] ) + 'r_{ s}$'
    if label_redshift:
      info_label = r'$z=' + '{:.3f}'.format( self.ptracks.redshift.iloc[slices] ) + '$, '+ info_label
    if label_galaxy_cut or label_redshift:
      ax.annotate( s=info_label, xy=(1.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,
        ha='right' )

    # Add axis labels
    if add_x_label:
      if x_label is default:
        x_label = x_key
      ax.set_xlabel( x_label, fontsize=label_fontsize )
    if add_y_label:
      if y_label is default:
        y_label = y_key
      ax.set_ylabel( y_label, fontsize=label_fontsize )

    # Limits
    ax.set_xlim( x_range )
    ax.set_ylim( y_range )

    # Set tick parameters
    if tick_param_args is not default:
      ax.tick_params( **tick_param_args )

    # Save the file
    if out_dir is not None:
      save_file = '{}_{:03d}.png'.format( self.label, self.ptracks.snum[slices] )
      gen_plot.save_fig( out_dir, save_file, fig=fig, dpi=75 )

      plt.close()

  ########################################################################
  # Generic Plotting Methods
  ########################################################################

  def panel_plot( self,
    panel_plotting_method_str,
    defaults,
    variations,
    slices = default,
    plot_label = default,
    outline_plot_label = False,
    label_galaxy_cut = True,
    label_redshift = True,
    label_fontsize = 24,
    subplot_label_args = { 'xy' : (0.075, 0.88), 'xycoords' : 'axes fraction', 'fontsize' : 18, 'color' : 'w',  },
    subplot_spacing_args = { 'hspace' : 0.0001, 'wspace' : 0.0001, },
    out_dir = None,
    ):
    '''
    Make a multi panel plot of the type of your choosing.
    Note: Currently only compatible with a four panel plot.

    Args:
      panel_plotting_method_str (str) : What type of plot to make.
      defaults (dict) : Default arguments to pass to panel_plotting_method.
      variations (dict of dicts) : Differences in plotting arguments per subplot.
      slices (slice) : What slices to select. By default, this doesn't pass any slices argument to panel_plotting_method
      plot_label (str or dict) : What to label the plot with. By default, uses self.label.
        Can also pass a dict of full args.
      outline_plot_label (bool) : If True, add an outline around the plot label.
      label_galaxy_cut (bool) : If true, add a label that indicates how the galaxy was defined.
      label_redshift (bool) : If True, add a label indicating the redshift.
      label_fontsize (int) : Fontsize for the labels.
      subplot_label_args (dict) : Label arguments to pass to each subplot for the label for the subplot.
        The actual label string itself corresponds to the keys in variations.
      subplot_spacing_args (dict) : How to space the subplots.
      out_dir (str) : If given, where to save the file.
    '''

    fig = plt.figure( figsize=(10,9), facecolor='white', )
    ax = plt.gca()

    fig.subplots_adjust( **subplot_spacing_args )

    if slices is not default:
      defaults['slices'] = slices

    plotting_kwargs = utilities.dict_from_defaults_and_variations( defaults, variations )

    # Setup axes
    gs = gridspec.GridSpec(2, 2)
    axs = []
    axs.append( plt.subplot( gs[0,0] ) )
    axs.append( plt.subplot( gs[0,1] ) )
    axs.append( plt.subplot( gs[1,0] ) )
    axs.append( plt.subplot( gs[1,1] ) )

    # Setup arguments further
    for i, key in enumerate( plotting_kwargs.keys() ):
      ax_kwargs = plotting_kwargs[key]

      ax_kwargs['ax'] = axs[i]

      # Subplot label args
      this_subplot_label_args = subplot_label_args.copy()
      this_subplot_label_args['s'] = key
      ax_kwargs['plot_label'] = this_subplot_label_args

      if ax_kwargs['add_colorbar']:
        ax_kwargs['colorbar_args'] = { 'fig_or_ax' : fig, 'ax_location' : [0.9, 0.125, 0.03, 0.775 ],  }

      # Clean up interior axes
      ax_tick_parm_args = ax_kwargs['tick_param_args'].copy()
      if i == 0:
        ax_kwargs['add_x_label'] = False
        ax_tick_parm_args['labelbottom'] = False
      if i == 1:
        ax_kwargs['add_x_label'] = False
        ax_tick_parm_args['labelbottom'] = False
        ax_kwargs['add_y_label'] = False
        ax_tick_parm_args['labelleft'] = False
      elif i == 3:
        ax_kwargs['add_y_label'] = False
        ax_tick_parm_args['labelleft'] = False
      ax_kwargs['tick_param_args'] = ax_tick_parm_args

    # Actual panel plots
    panel_plotting_method = getattr( self, panel_plotting_method_str )
    for key in plotting_kwargs.keys():
      panel_plotting_method( **plotting_kwargs[key] )

    # Main axes labels
    # Plot label
    if plot_label is default:
      plt_label = axs[0].annotate( s=self.label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
    elif isinstance( plot_label, str ):
      plt_label = axs[0].annotate( s=plot_label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
    elif isinstance( plot_label, dict ):
      plt_label = axs[0].annotate( **plot_label )
    else:
      raise Exception( 'Unrecognized plot_label arguments, {}'.format( plot_label ) )
    if outline_plot_label:
      plt_label.set_path_effects([ path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal() ])

    # Upper right label (info label)
    info_label = ''
    if label_galaxy_cut:
      info_label = r'$r_{ \rm cut } = ' + '{:.3g}'.format( self.galids.parameters['galaxy_cut'] ) + 'r_{ s}$'
    if label_redshift:
      ind = defaults['slices']
      info_label = r'$z=' + '{:.3f}'.format( self.ptracks.redshift.iloc[ind] ) + '$, '+ info_label
    if label_galaxy_cut or label_redshift:
      axs[1].annotate( s=info_label, xy=(1.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,
        ha='right' )

    # Save the file
    if out_dir is not None:
      save_file = '{}_{:03d}.png'.format( self.label, self.ptracks.snum[slices] )
      gen_plot.save_fig( out_dir, save_file, fig=fig )

      plt.close()

  ########################################################################

  def make_multiple_plots( self,
    plotting_method_str,
    iter_args_key,
    iter_args,
    n_processors = 1,
    out_dir = None,
    make_movie = False,
    clear_data = False,
    *args, **kwargs ):
    '''Make multiple plots of a selected type. *args and **kwargs are passed to plotting_method_str.

    Args:
      plotting_method_str (str) : What plotting method to use.
      iter_args_key (str) : The name of the argument to iterate over.
      iter_args (list) : List of argument values to change.
      n_processors (int) : Number of processors to use. Should only be used when saving the data.
      out_dir (str) : Where to save the data.
      make_movie (bool) : Make a movie out of the plots, if True.
      clear_data (bool) : If True, clear memory of the data after making the plots.
    '''
    
    plotting_method = getattr( self, plotting_method_str )

    if out_dir is not None:
      out_dir = os.path.join( out_dir, self.label )

    def plotting_method_wrapper( process_args ):

      used_out_dir, used_args, used_kwargs = process_args

      plotting_method( out_dir=used_out_dir, *used_args, **used_kwargs )

      del used_out_dir, used_args, used_kwargs

      return

    all_process_args = []
    for iter_arg in iter_args:
      process_kwargs = dict( kwargs )
      process_kwargs[iter_args_key] = iter_arg
      all_process_args.append( ( out_dir, args, process_kwargs ) )

    if n_processors > 1:
      # For safety, make sure we've loaded the data already
      self.ptracks, self.galids, self.classifications

      mp_utils.parmap( plotting_method_wrapper, all_process_args, n_processors=n_processors, return_values=False )
    else:
      for i, iter_arg in enumerate( iter_args ):
        plotting_method_wrapper( all_process_args[i] )

    if make_movie:
      gen_plot.make_movie( out_dir, '{}_*.png'.format( self.label ), '{}.mp4'.format( self.label ), )

    if clear_data:
      del self.ptracks
      del self.galids
      del self.classifications
