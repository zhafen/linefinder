#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

import plot_worldlines
import analyze_worldlines as a_worldlines

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
      defaults (dict) : Set of default arguments for loading worldline data.
      variations (dict of dicts) : Labels and differences in arguments to be passed to Worldlines
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
  # Plotting Methods
  ########################################################################

  def plot_w_set_same_axis( self,
    plotting_method,
    *args, **kwargs ):

    fig = plt.figure( figsize=(11,5), facecolor='white' )
    ax = plt.gca()

    # The plot itself
    getattr( self, plotting_method )( ax=ax, *args, **kwargs )

    ax.legend(loc='upper center', prop={'size':14.5}, fontsize=20)

  ########################################################################

  def plot_classification_bar_same_axis( self,
    kwargs=default,
    ind = 0,
    width = 0.5,
    data_order = default,
    **default_kwargs ):

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

    ax.set_ylabel( r'$f(M_{\star})$', fontsize=22 )

    redshift = self[key].data_object.get_data( 'redshift' )[ind]
    title_string = r'$z=' + '{:.3f}'.format( redshift ) + '$'
    ax.set_title( title_string, fontsize=22, )
    
    ax.legend(prop={'size':14.5}, ncol=5, loc=(0.,-0.2), fontsize=20)
