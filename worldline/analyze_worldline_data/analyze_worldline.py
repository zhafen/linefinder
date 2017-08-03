#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

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

    # Load the data
    self.ptrack_data = analyze_ptrack.PTrackData( tracking_dir, tag )
    self.galfind_data = analyze_galfind.GalFindData( tracking_dir, tag )
    self.classified_data = analyze_classified.ClassifiedData( tracking_dir, tag )
