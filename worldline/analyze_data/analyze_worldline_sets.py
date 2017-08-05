#!/usr/bin/env python
'''Tools for loading in multiple worldline data sets, for comparison

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

########################################################################
########################################################################

class WorldlinesSet( object ):
  '''Wrapper for multiple WorldlineData classes.
  '''

  def __init__( self, defaults, modified ):
    '''
    Args:
      defaults (dict) : Set of default arguments for loading worldline data.
      differences (dict of dicts) : Differences between different sets to load.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

