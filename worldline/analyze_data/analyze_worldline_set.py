#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import analyze_worldlines

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

      worldlines_d[key] = analyze_worldlines.Worldlines( label=key, **kwargs )

    super( WorldlineSet, self ).__init__( worldlines_d )
