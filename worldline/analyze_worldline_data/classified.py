#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os

########################################################################
########################################################################

class ClassifiedData( object ):
  '''Loads and analyzes data created by classify.py

  There's nothing actually classified (in a TOP SECRET sense) in classified data, by the way.
  It's just an (un)fortunate naming coincidence.
  Classified data simply contains data that has been classified (in a scientific sense) into different categories!
  I could call it something else, but this seems more fun.
  What could go wrong?
  '''

  def __init__( self, tracking_dir, tag ):
    '''
    Args:
      tracking_dir (str) : Data directory for the classified data
      tag (str) : Identifying tag for the data to load.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    # Open the file
    classified_data_filepath = os.path.join( tracking_dir, 'classified_{}.hdf5'.format( tag ) )
    f = h5py.File( classified_data_filepath, 'r' )

    # Store the data
    self.data = {}
    for key in f.keys():
      self.data[key] = f[key][...]

    # Store the data attributes
    self.data_attrs = {}
    for key in f.attrs.keys():
      self.data_attrs[key] = f.attrs[key]

  ########################################################################

  def calc_base_fractions( self ):
    '''Get first order results of the classified data, in the form of fractions of each main type of classified data, relative to the total.

    Returns:
      base_fractions (dict) : Contains the fractions for each main type of classified data, evaluated at the final snapshot
    '''

    base_fractions = {}

    # Get the total number of particles
    n_classified = float( self.data['is_pristine'].size )

    base_fractions['fresh accretion'] = float( self.data['is_pristine'].sum() )/n_classified
    base_fractions['merger'] = float( self.data['is_merger'].sum() )/n_classified
    base_fractions['intergalactic transfer'] = float( self.data['is_mass_transfer'].sum() )/n_classified

    # For wind, only evaluate at the last snapshot
    base_fractions['wind'] = float( self.data['is_wind'][:,0].sum() )/n_classified

    return base_fractions
