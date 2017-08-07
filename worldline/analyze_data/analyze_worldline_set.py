#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import collections
import numpy as np

import analyze_worldlines

########################################################################
########################################################################

class WorldlineSet( collections.Mapping ):
  '''Container for multiple Worldlines classes.
  '''

  def __init__( self, defaults, variations ):
    '''
    Args:
      defaults (dict) : Set of default arguments for loading worldline data.
      variations (dict of dicts) : Labels and differences in arguments to be passed to Worldlines
    '''

    # Load the worldline sets
    self._worldlines = {}
    for key in variations.keys():
      
      kwargs = dict( defaults )
      for var_key in variations[key].keys():
        kwargs[var_key] = variations[key][var_key]

      self._worldlines[key] = analyze_worldlines.Worldlines( **kwargs )

  ########################################################################

  def __getitem__( self, key ):
    
    return self._worldlines[key]

  ########################################################################

  def __iter__( self ):
    return iter( self._worldlines )

  ########################################################################

  def __len__( self ):
    return len( self._worldlines )

  ########################################################################

  def __getattr__( self, attr ):

    results = {}
    for key in self.keys():
      results[key] = getattr( self._worldlines[key], attr )

    return results
