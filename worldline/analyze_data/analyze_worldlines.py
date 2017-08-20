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

import galaxy_diver.analyze_data.generic_data as generic_data
import galaxy_diver.plot_data.ahf as plot_ahf
import galaxy_diver.plot_data.plotting as gen_plot
import galaxy_diver.plot_data.pu_colormaps as pu_cm
import galaxy_diver.utils.mp_utils as mp_utils
import galaxy_diver.utils.utilities as utilities

import analyze_ids
import analyze_ptracks
import analyze_galids
import analyze_classifications
import analyze_events

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class Worldlines( generic_data.GenericData ):
  '''Wrapper for analysis of all worldline data products. It loads data in on-demand.
  '''

  def __init__( self,
    data_dir,
    tag,
    ids_tag = default,
    ptracks_tag = default,
    galids_tag = default,
    classifications_tag = default,
    events_tag = default,
    label = default,
    **kwargs ):
    '''
    Args:
      data_dir (str) : Data directory for the classified data
      tag (str) : Identifying tag for the data to load.
      ids_tag (str) : Identifying tag for ids data. Defaults to tag.
      ptracks_tag (str) : Identifying tag for ptracks data. Defaults to tag.
      galids_tag (str) : Identifying tag for galids data. Defaults to tag.
      classifications_tag (str) : Identifying tag for classifications data. Defaults to tag.
      events_tag (str) : Identifying tag for events data. Defaults to tag.
      label (str) : Identifying label for the worldlines. Defaults to tag.
    '''

    if ids_tag is default:
      ids_tag = tag
    if ptracks_tag is default:
      ptracks_tag = tag
    if galids_tag is default:
      galids_tag = tag
    if classifications_tag is default:
      classifications_tag = tag
    if events_tag is default:
      events_tag = tag

    if label is default:
      label = tag

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    self.ptracks_kwargs = dict( kwargs )

    data_masker = WorldlineDataMasker( self )

    super( Worldlines, self ).__init__( data_masker=data_masker, **kwargs )

  ########################################################################
  # Properties for loading data on the fly
  ########################################################################

  @property
  def ids( self ):

    if not hasattr( self, '_ids' ):
      self._ids = analyze_ids.IDs( self.data_dir, self.ids_tag, )

    return self._ids

  @ids.deleter
  def ids( self ):
    del self._ids

  ########################################################################

  @property
  def ptracks( self ):

    if not hasattr( self, '_ptracks' ):
      self._ptracks = analyze_ptracks.PTracks( self.data_dir, self.ptracks_tag, store_ahf_reader=True,
                                               **self.ptracks_kwargs )

    return self._ptracks

  @ptracks.deleter
  def ptracks( self ):
    del self._ptracks

  ########################################################################

  @property
  def galids( self ):

    if not hasattr( self, '_galids' ):
      self._galids = analyze_galids.GalIDs( self.data_dir, self.galids_tag )

    return self._galids

  @galids.deleter
  def galids( self ):
    del self._galids

  ########################################################################

  @property
  def classifications( self ):

    if not hasattr( self, '_classifications' ):
      self._classifications = analyze_classifications.Classifications( self.data_dir, self.classifications_tag )

    return self._classifications

  @classifications.deleter
  def classifications( self ):
    del self._classifications

  ########################################################################

  @property
  def events( self ):

    if not hasattr( self, '_events' ):
      self._events = analyze_events.Events( self.data_dir, self.events_tag )

    return self._events

  @events.deleter
  def events( self ):
    del self._events

  ########################################################################

  @property
  def base_data_shape( self ):

    return self.ptracks.base_data_shape

  ########################################################################

  @property
  def n_snaps( self ):

    if not hasattr( self, '_n_snaps' ):
      self._n_snaps = self.ptracks.base_data_shape[1]

    return self._n_snaps

  ########################################################################

  @property
  def n_particles( self ):

    if not hasattr( self, '_n_particles' ):
      self._n_particles = self.ptracks.base_data_shape[0]

    return self._n_particles

  ########################################################################

  @property
  def redshift( self ):

    if not hasattr( self, '_redshift' ):
      self._redshift = self.ptracks.redshift

    return self._redshift

  ########################################################################
  # Display Information
  ########################################################################

  def get_parameters( self ):

    parameters = {}
    for data in [ 'ids', 'ptracks', 'galids', 'classifications' ]:

      parameters[data] = getattr( self, data ).parameters

    return parameters

  ########################################################################
  # Get Data
  ########################################################################

  def get_data( self, data_key, *args, **kwargs ):
    '''Get data. Usually just get it from ptracks. args and kwargs are passed to self.ptracks.get_data()

    Args:
      data_key (str) : What data to get?

    Returns:
      data (np.ndarray) : Array of data.
    '''

    data = self.ptracks.get_data( data_key, *args, **kwargs )

    return data

  ########################################################################
  # Plotting
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

########################################################################
########################################################################

class WorldlineDataMasker( generic_data.DataMasker ):
  '''Data masker for worldline data.'''

  def __init__( self, worldlines ):

    super( WorldlineDataMasker, self ).__init__( worldlines )

  ########################################################################

  def get_mask( self,
    mask =default,
    classification = None,
    mask_after_first_acc = False,
    mask_before_first_acc = False,
    preserve_mask_shape = False,
    *args, **kwargs ):
    '''Get a mask for the data.

    Args:
      mask (np.array) : Mask to apply to the data. If default, use the masks stored in self.masks (which defaults to
        empty).
      classification (str) : If provided, only select particles that meet this classification, as given in
        self.data_object.classifications.data
      mask_after_first_acc (bool) : If True, only select particles above first accretion.
      mask_before_first_acc (bool) : If True, only select particles after first accretion.
      preserve_mask_shape (bool) : If True, don't tile masks that are single dimensional, and one per particle.

    Returns:
      mask (bool np.ndarray) : Mask from all the combinations.
    '''

    used_masks = []
    if mask is default:
      used_masks += self.masks
    else:
      
      # Tile mask if it's single-dimensional
      if ( not preserve_mask_shape ) and ( mask.shape == ( self.data_object.n_particles, ) ):
        mask = np.tile( mask, (self.data_object.n_snaps, 1 ) ).transpose()

      used_masks.append( mask )

    if classification is not None:
      cl_mask = np.invert( self.data_object.classifications.data[classification] ) 
      if classification != 'is_wind':
        cl_mask = np.tile( cl_mask, (self.data_object.n_snaps, 1) ).transpose()
      used_masks.append( cl_mask )

    if mask_after_first_acc or mask_before_first_acc:

      assert not ( mask_after_first_acc and mask_before_first_acc ), "Attempted to mask both before and after first acc."

      redshift_tiled = np.tile( self.data_object.redshift, (self.data_object.n_particles, 1) )
      redshift_first_acc_tiled = np.tile( self.data_object.events.data['redshift_first_acc'],
                                          (self.data_object.n_snaps, 1) ).transpose()
      if mask_after_first_acc:
        first_acc_mask = redshift_tiled <= redshift_first_acc_tiled
      elif mask_before_first_acc:
        first_acc_mask = redshift_tiled > redshift_first_acc_tiled
      used_masks.append( first_acc_mask )

    mask = np.any( used_masks, axis=0, keepdims=True )[0]

    return mask

  ########################################################################

  def get_masked_data( self,
    data_key,
    mask=default,
    classification=None,
    mask_after_first_acc=False,
    mask_before_first_acc=False,
    preserve_mask_shape=False,
    *args, **kwargs ):
    '''Get masked worldline data. Extra arguments are passed to the ParentClass' get_masked_data.

    Args:
      data_key (str) : Data to get.
      mask (np.array) : Mask to apply to the data. If default, use the masks stored in self.masks (which defaults to
        empty).
      classification (str) : If provided, only select particles that meet this classification, as given in
        self.data_object.classifications.data
      mask_after_first_acc (bool) : If True, only select particles above first accretion.
      mask_before_first_acc (bool) : If True, only select particles after first accretion.

    Returns:
      masked_data (np.array) : Flattened array of masked data.
    '''

    used_mask = self.get_mask(
      mask=mask,
      classification=classification,
      mask_after_first_acc=mask_after_first_acc,
      mask_before_first_acc=mask_before_first_acc,
      preserve_mask_shape=preserve_mask_shape,
    )

    masked_data = super( WorldlineDataMasker, self ).get_masked_data( data_key, mask=used_mask, *args, **kwargs )

    return masked_data

































