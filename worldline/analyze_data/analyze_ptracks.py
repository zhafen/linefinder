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

class PTracks( object ):
  '''Loads and analyzes data created by galaxy_find.py
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
    ptracks_filepath = os.path.join( tracking_dir, 'ptracks_{}.hdf5'.format( tag ) )
    with h5py.File( ptracks_filepath, 'r' ) as f:

      # Store the data
      self.data = {}
      for key in f.keys():
        if key != 'parameters':
          self.data[key] = f[key][...]

      # Store the data attributes
      self.data_attrs = {}
      for key in f.attrs.keys():
        self.data_attrs[key] = f.attrs[key]

      # Store the parameters
      self.parameters = {}
      param_grp = f['parameters']
      for key in param_grp.attrs.keys():
        self.parameters[key] = param_grp.attrs[key]

