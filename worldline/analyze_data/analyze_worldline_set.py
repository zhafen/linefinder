#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import analyze_worldlines

########################################################################
########################################################################

class WorldlineSet( object ):
  '''Wrapper for multiple Worldlines classes.
  '''

  def __init__( self, defaults, variations ):
    '''
    Args:
      defaults (dict) : Set of default arguments for loading worldline data.
      variations (dict of dicts) : Labels and differences in arguments to be passed to Worldlines
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    # Load the worldline sets
    self.worldlines = {}
    for key in variations.keys():
      
      kwargs = dict( defaults )
      for var_key in variations[key].keys():
        kwargs[var_key] = variations[key][var_key]

      self.worldlines[key] = analyze_worldlines.Worldlines( **kwargs )
