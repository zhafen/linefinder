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

  def plot_w_set_same_axis( self,
    plotting_method,
    *args, **kwargs ):

    fig = plt.figure( figsize=(11,5), facecolor='white' )
    ax = plt.gca()

    # The plot itself
    getattr( self, plotting_method )( ax=ax, *args, **kwargs )

    ax.legend(loc='upper center', prop={'size':14.5}, fontsize=20)
