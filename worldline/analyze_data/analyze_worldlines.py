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
import matplotlib.patheffects as path_effects

import galaxy_diver.analyze_data.generic_data as generic_data
import galaxy_diver.plot_data.ahf as plot_ahf
import galaxy_diver.plot_data.plotting as gen_plot
import galaxy_diver.plot_data.pu_colormaps as pu_cm
import galaxy_diver.utils.mp_utils as mp_utils

import analyze_ptracks
import analyze_galids
import analyze_classifications

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class Worldlines( object ):
  '''Wrapper for analysis of all worldline data products. It loads data in on-demand.
  '''

  def __init__( self, data_dir, tag, label=None, **kwargs ):
    '''
    Args:
      data_dir (str) : Data directory for the classified data
      tag (str) : Identifying tag for the data to load.
      label (str) : Identifying label for the worldlines.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

  ########################################################################
  # Properties for loading data on the fly
  ########################################################################

  @property
  def ptracks( self ):

    if not hasattr( self, '_ptracks' ):
      self._ptracks = analyze_ptracks.PTracks( self.data_dir, self.tag, store_ahf_reader=True, **self.kwargs )

    return self._ptracks

  ########################################################################

  @property
  def galids( self ):

    if not hasattr( self, '_galids' ):
      self._galids = analyze_galids.GalIDs( self.data_dir, self.tag )

    return self._galids

  ########################################################################

  @property
  def classifications( self ):

    if not hasattr( self, '_classifications' ):
      self._classifications = analyze_classifications.Classifications( self.data_dir, self.tag )

    return self._classifications

  ########################################################################

  @property
  def data_masker( self ):
  
    if not hasattr( self, '_data_masker' ):
      self._data_masker = WorldlineDataMasker( self )

    return self._data_masker

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
    for data in [ 'ptracks', 'galids', 'classifications' ]:

      parameters[data] = getattr( self, data ).parameters

    return parameters

  ########################################################################
  # Get Data
  ########################################################################

  def get_masked_data(  self, *args, **kwargs ):
    '''Wrapper for masking data.'''

    return self.data_masker.get_masked_data( *args, **kwargs )

  ########################################################################
  # Plotting
  ########################################################################

  def plot_hist_2d( self,
                    x_key, y_key,
                    slices,
                    x_range=default, y_range=default,
                    n_bins=128,
                    vmin=None, vmax=None,
                    x_label=default, y_label=default,
                    plot_halos=False,
                    label_plot=True,
                    label_redshift=True,
                    out_dir=None,
                    *args, **kwargs ):
    '''Make a 2D histogram of the data. Extra arguments are passed to get_masked_data.
    Args:
      x_key, y_key (str) : Data keys to plot.
      slices (int or tuple of slices) : How to slices the data.
      x_range, y_range ( (float, float) ) : Histogram edges. If default, all data is enclosed. If list, set manually.
        If float, is +- x_range*length scale at that snapshot.
      n_bins (int) : Number of bins in the histogram.
      vmin, vmax (float) : Limits for the colorbar.
      x_label, ylabel_ (str) : Axes labels.
      plot_halos (bool) : Whether or not to plot merger tree halos on top of the histogram.
        Only makes sense for when dealing with positions.
      label_plot (bool) : If True, label with self.label.
      label_redshift (bool) : If True, add a label indicating the redshift.
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
    fig = plt.figure( figsize=(10,9), facecolor='white' )
    ax = plt.gca()

    im = ax.imshow( np.log10( hist2d ).transpose(), cmap=pu_cm.magma, interpolation='nearest',\
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], \
                    origin='low', aspect='auto', vmin=vmin, vmax=vmax, )

    # Add a colorbar
    cbar = gen_plot.add_colorbar( ax, im, method='ax' )
    cbar.ax.tick_params( labelsize=20 )

    # Halo Plot
    if plot_halos:
      ahf_plotter = plot_ahf.AHFPlotter( self.ptracks.ahf_reader )
      snum = self.ptracks.ahf_reader.mtree_halos[0].index[slices]
      ahf_plotter.plot_halos_snapshot( snum, ax, hubble_param=self.ptracks.data_attrs['hubble'],
        radius_fraction=self.galids.parameters['galaxy_cut'] )
      assert self.galids.parameters['length_scale'] == 'r_scale'
      info_label = r'$r_{ \rm cut } = ' + '{:.3g}'.format( self.galids.parameters['galaxy_cut'] ) + 'r_{ s}$'

    # Labels
    if label_plot:
      ax.annotate( s=self.label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=24,  )
    if label_redshift:
      info_label = r'$z=' + '{:.3f}'.format( self.ptracks.redshift.iloc[slices] ) + '$, '+ info_label
    if plot_halos or label_redshift:
      ax.annotate( s=info_label, xy=(1.,1.0225), xycoords='axes fraction', fontsize=24,
        ha='right' )

    # Add labels
    if x_label is default:
      x_label = x_key
    if y_label is default:
      y_label = y_key
    ax.set_xlabel( x_label, fontsize=24 )
    ax.set_ylabel( y_label, fontsize=24 )

    # Limits
    ax.set_xlim( x_range )
    ax.set_ylim( y_range )

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
                           n_processors=1,
                           out_dir=None,
                           make_movie=False,
                           *args, **kwargs ):
    '''Make multiple plots of a selected type. *args and **kwargs are passed to plotting_method_str.

    Args:
      plotting_method_str (str) : What plotting method to use.
      iter_args_key (str) : The name of the argument to iterate over.
      iter_args (list) : List of argument values to change.
      n_processors (int) : Number of processors to use. Should only be used when saving the data.
      out_dir (str) : Where to save the data.
      make_movie (bool) : Make a movie out of the plots, if True.
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

########################################################################
########################################################################

class WorldlineDataMasker( generic_data.DataMasker ):
  '''Data masker for worldline data.'''

  def __init__( self, worldlines ):

    self.worldlines = worldlines

    super( WorldlineDataMasker, self ).__init__( self.worldlines.ptracks )

  ########################################################################

  def get_masked_data( self,
                       data_key,
                       mask=default,
                       classification=None,
                       mask_after_first_acc=False,
                       *args, **kwargs ):
    '''Get masked worldline data. Extra arguments are passed to the ParentClass' get_masked_data.

    Args:
      data_key (str) : Data to get.
      mask (np.array) : Mask to apply to the data. If default, use the masks stored in self.masks (which defaults to
        empty).
      classification (str) : If provided, only select particles that meet this classification, as given in
        self.worldlines.classifications.data
      mask_after_first_acc (bool) : If True, only select particles above first accretion.

    Returns:
      masked_data (np.array) : Flattened array of masked data.
    '''

    used_masks = []
    if mask is default:
      used_masks += self.masks
    else:
      used_masks.append( mask )

    if classification is not None:
      cl_mask = np.invert( self.worldlines.classifications.data[classification] ) 
      cl_mask = np.tile( cl_mask, (self.worldlines.n_snaps, 1) ).transpose()
      used_masks.append( cl_mask )

    if mask_after_first_acc:
      redshift_tiled = np.tile( self.worldlines.redshift, (self.worldlines.n_particles, 1) )
      redshift_first_acc_tiled = np.tile( self.worldlines.classifications.data['redshift_first_acc'],
                                          (self.worldlines.n_snaps, 1) ).transpose()
      after_first_acc_mask = redshift_tiled <= redshift_first_acc_tiled
      used_masks.append( after_first_acc_mask )

    used_mask = np.any( used_masks, axis=0, keepdims=True )[0]

    masked_data = super( WorldlineDataMasker, self ).get_masked_data( data_key, mask=used_mask, *args, **kwargs )

    return masked_data

































