#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os

import galaxy_diver.analyze_data.simulation_data as simulation_data

########################################################################
########################################################################

class PTracks( simulation_data.TimeData ):
  '''Loads and analyzes data created by galaxy_find.py
  '''

  def __init__( self, data_dir, tag, ahf_index=None, *args, **kwargs ):
    '''
    Args:
      data_dir (str) : Data directory for the classified data
      tag (str) : Identifying tag for the data to load.
      ahf_index (str or int) : Index to use for AHF data.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    # Open the file
    ptracks_filepath = os.path.join( data_dir, 'ptracks_{}.hdf5'.format( tag ) )
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

    # Reorganize data to match with formatting in TimeData
    self.data['P'] = np.rollaxis( self.data['P'], 2 )
    self.data['V'] = np.rollaxis( self.data['V'], 2 )

    super( PTracks, self ).__init__( data_dir, snum=self.data['snum'], ahf_index=ahf_index, *args, **kwargs )

  ########################################################################

  @property
  def snum( self ):
    
    return self.data['snum']

  ########################################################################

  @property
  def hubble_param( self ):
    
    return self.data_attrs['hubble']
