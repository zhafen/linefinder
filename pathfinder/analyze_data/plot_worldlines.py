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
import matplotlib.transforms as transforms

import galaxy_diver.plot_data.generic_plotter as generic_plotter
import galaxy_diver.plot_data.ahf as plot_ahf
import galaxy_diver.plot_data.plotting as gen_plot
import galaxy_diver.plot_data.pu_colormaps as pu_cm
import galaxy_diver.utils.mp_utils as mp_utils
import galaxy_diver.utils.utilities as utilities

import pathfinder.utils.presentation_constants as p_constants

import analyze_worldlines

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class WorldlinesPlotter( generic_plotter.GenericPlotter ):

  def plot_classification_bar( self,
    x_pos,
    ind = 0,
    ax = default,
    width = 0.5,
    add_label = False,
    ):

    # Plot
    if ax is default:
      fig = plt.figure( figsize=(11,5), facecolor='white' )
      ax = plt.gca()

    classification_values = self.data_object.get_categories_stellar_mass_fraction( ind=ind )

    bar_start = 0.
    for i, key in enumerate( p_constants.CLASSIFICATION_LIST_A ):

      if add_label:
        label = p_constants.CLASSIFICATION_LABELS[key]
      else:
        label = None

      value = classification_values[key]
      ax.bar( x_pos, value, width, bottom=bar_start, color=p_constants.CLASSIFICATION_COLORS[key],
        label=label, )

      bar_start += value

  ########################################################################

  def plot_classification_values( self,
    ax = default,
    label = default,
    color = default,
    pointsize = 3000,
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

  def plot_dist_hist( self,
    ax,
    *args, **kwargs ):

    if self.data_object.galids.parameters['length_scale'] == 'r_scale':
      x_label = r'Distance to Nearest Satellite ($r_{ \rm scale, sat }$)'
    self.plot_hist( ax=ax, x_label=x_label, *args, **kwargs )

    # Add a bar indicating our radial cut
    trans = transforms.blended_transform_factory( ax.transData, ax.transAxes )
    r_cut = self.data_object.galids.parameters['galaxy_cut']
    ax.plot( [ r_cut, ]*2, [0, 1], color='black', linewidth=3, linestyle='--', transform=trans )


