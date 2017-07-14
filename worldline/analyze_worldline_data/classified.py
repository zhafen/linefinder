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
      tag (str): Identifying tag for the data to load.
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
      self.data[key] = f[key]

    # Store the data attributes
    self.data_attrs = {}
    for key in f.attrs.keys():
      self.data_attrs[key] = f.attrs[key]

  ########################################################################

