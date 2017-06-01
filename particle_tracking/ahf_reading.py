#!/usr/bin/env python
'''Tools for reading ahf output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import os
import pandas as pd
import string

########################################################################
########################################################################

class AHFReader( object ):
  '''Read AHF data.'''

  def __init__( self, sdir ):
    '''Initializes.

    Args:
      sdir (str): Simulation directory to load the AHF data from.
    '''

    self.sdir = sdir

  ########################################################################

  def get_ahf_halos( self, snum ):
    '''Get *.AHF_halos file for a particular snapshot.

    Args:
      snum (int): Snapshot number to load.

    Returns:
      self.ahf_halos (pd.DataFrame): Dataframe containing the requested data.
    '''

    # Load the data
    ahf_halos_filename = 'snap{:03d}Rpep..z0.000.AHF_halos'.format( snum )
    ahf_halos_path = os.path.join( self.sdir, ahf_halos_filename )
    self.ahf_halos = pd.read_csv( ahf_halos_path, sep='\t', index_col=0 )

    # Delete a column that shows up as a result of formatting
    del self.ahf_halos[ 'Unnamed: 92' ]

    # Remove the annoying parenthesis at the end of each label.
    self.ahf_halos.columns = [ string.split( label, '(' )[0] for label in list( self.ahf_halos ) ]

    return self.ahf_halos
