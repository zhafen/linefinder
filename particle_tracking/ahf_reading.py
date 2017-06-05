#!/usr/bin/env python
'''Tools for reading ahf output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import os
import pandas as pd
import string

########################################################################
########################################################################

class AHFReader( object ):
  '''Read AHF data.
  Note! All positions are in comoving coordinates, and everything has 1/h's sprinkled throughout.
  '''

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

    # If the data's already loaded, don't load it again.
    if hasattr( self, 'ahf_halos' ):
      return self.ahf_halos

    # Load the data
    ahf_halos_path = self.get_ahf_filepath( snum, 'AHF_halos' )
    self.ahf_halos = pd.read_csv( ahf_halos_path, sep='\t', index_col=0 )

    # Delete a column that shows up as a result of formatting
    del self.ahf_halos[ 'Unnamed: 92' ]

    # Remove the annoying parenthesis at the end of each label.
    self.ahf_halos.columns = [ string.split( label, '(' )[0] for label in list( self.ahf_halos ) ]

    # Rename the index to a more suitable name, without the '#' and the (1)
    self.ahf_halos.index.names = ['ID']

    return self.ahf_halos

  ########################################################################

  def get_ahf_mtree_idx( self, snum ):
    '''Get *.AHF_mtree_idx file for a particular snapshot.

    Args:
      snum (int): Snapshot number to load.

    Returns:
      self.ahf_mtree_idx (pd.DataFrame): Dataframe containing the requested data.
    '''

    # If the data's already loaded, don't load it again.
    if hasattr( self, 'ahf_mtree_idx' ):
      return self.ahf_mtree_idx

    # Load the data
    ahf_mtree_idx_path = self.get_ahf_filepath( snum, 'AHF_mtree_idx' )
    self.ahf_mtree_idx = pd.read_csv( ahf_mtree_idx_path, delim_whitespace=True, names=['HaloID(1)', 'HaloID(2)'], skiprows=1  )

    return self.ahf_mtree_idx

  ########################################################################

  def get_ahf_filepath( self, snum, ahf_file_type ):
    '''Get the filepath for a specified type of AHF file.

    Args:
      snum (int): Snapshot number to load.
      ahf_file_type (str): Can be AHF_halos or AHF_mtree_idx.

    Returns:
      ahf_filepath (str): The filepath to the specified file.
    '''

    # Load the data
    ahf_filename = 'snap{:03d}Rpep..z*.*.{}'.format( snum, ahf_file_type )
    ahf_filepath_unexpanded = os.path.join( self.sdir, ahf_filename )
    possible_filepaths = glob.glob( ahf_filepath_unexpanded )
    if len( possible_filepaths ) > 1:
      raise Exception( 'Multiple possible *.{} files to load'.format( ahf_file_type ) )
    ahf_filepath = possible_filepaths[0]

    return ahf_filepath
