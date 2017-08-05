#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import analyze_ptrack
import analyze_galfind
import analyze_classified

########################################################################
########################################################################

class WorldlineData( object ):
  '''Wrapper for analysis of all worldline data products.
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

  ########################################################################

  @property
  def ptrack_data( self ):

    if not hasattr( self, '_ptrack_data' ):
      self._ptrack_data = analyze_ptrack.PtrackData( self.tracking_dir, self.tag )

    return self._ptrack_data

  ########################################################################

  @property
  def galfind_data( self ):

    if not hasattr( self, '_galfind_data' ):
      self._galfind_data = analyze_galfind.GalfindData( self.tracking_dir, self.tag )

    return self._galfind_data

  ########################################################################

  @property
  def classified_data( self ):

    if not hasattr( self, '_classified_data' ):
      self._classified_data = analyze_classified.ClassifiedData( self.tracking_dir, self.tag )

    return self._classified_data

  ########################################################################


