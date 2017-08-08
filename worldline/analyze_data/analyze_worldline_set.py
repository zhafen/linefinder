#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import collections
import numpy as np

import analyze_worldlines

import galaxy_diver.utils.utilities as utilities

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
  # Methods for Being Dictionary-Like
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
  # Methods for Getting Data from Worldlines
  ########################################################################

  def __getattr__( self, attr, ):
    '''Replacement for default attribute retrieval. E.g. worldline_set.foo == worldline_set.__getattr__( 'foo' )
    Instead returns a dictionary containing the results of the attr for each Worldlines in the set.
    '''

    results = {}
    for key in self.keys():

      results[key] = utilities.deepgetattr( self._worldlines[key], attr )

    return results

  ########################################################################

  def get( self, method, *args, **kwargs ):
    '''Generic getter method for any method in analyze_worldlines.Worldlines.
    This applies the chosen method across the full set of Worldlines.

    Args:
      method (str) : The method of Worldlines you want to get out.
      *args, **kwargs : Arguments to supply to method.

    Returns:
      results (dict) : results[key] is equal to WorldlineSet[key].method( *args, **kwargs ).
    '''

    results = {}
    for key in self.keys():

      key_method = utilities.deepgetattr( self._worldlines[key], method )

      results[key] = key_method( *args, **kwargs )

    return results
