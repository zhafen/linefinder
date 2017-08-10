#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import galaxy_diver.plot_data.pu_colormaps as pu_cm
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects

import galaxy_diver.plot_data.plotting as gen_plot

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

  def __init__( self, data_dir, tag, **kwargs ):
    '''
    Args:
      data_dir (str) : Data directory for the classified data
      tag (str) : Identifying tag for the data to load.
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
      self._ptracks = analyze_ptracks.PTracks( self.data_dir, self.tag, **self.kwargs )

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
  # Display Information
  ########################################################################

  def get_parameters( self ):

    parameters = {}
    for data in [ 'ptracks', 'galids', 'classifications' ]:

      parameters[data] = getattr( self, data ).parameters

    return parameters

  ########################################################################
  # Plotting
  ########################################################################

  def plot_hist_2d( self,
                    x_key,
                    y_key,
                    ind,
                    x_label=default,
                    y_label=default,
                    n_bins=128 ):

    x_data = self.ptracks.get_processed_data( x_key )[:,ind]
    y_data = self.ptracks.get_processed_data( y_key )[:,ind]

    # Make the histogram
    hist2d, x_edges, y_edges = np.histogram2d( x_data, y_data, n_bins )

    # Plot
    fig = plt.figure( figsize=(7,6), facecolor='white' )
    ax = plt.gca()

    im = ax.imshow( np.log10( hist2d ).transpose(), cmap=pu_cm.magma, interpolation='nearest',\
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], \
                    origin='low', aspect='auto' )

    # Add a colorbar
    cbar = gen_plot.add_colorbar( ax, im, method='ax' )
    cbar.ax.tick_params( labelsize=20 )

    if x_label is default:
      x_label = x_key
    if y_label is default:
      y_label = y_key

    # Add labels
    ax.set_xlabel( x_label, fontsize=24 )
    ax.set_ylabel( y_label, fontsize=24 )
