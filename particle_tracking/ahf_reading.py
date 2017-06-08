#!/usr/bin/env python
'''Tools for reading ahf output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import numpy as np
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

  def get_mtree_halos( self, index=None ):
    '''Get halo files (e.g. halo_00000.dat) in a dictionary of pandas DataFrames.

    Args:
      index (str) : What type of index to use. Defaults to None, which raises an exception. You *must* choose an index, to avoid easy mistakes. Options are...
        'snum' : Indexes by snapshot number, starting at 600 and counting down. Only viable with snapshot steps of 1!!
        'int' : Index by integer.

    Returns:
      self.mtree_halos (dict of pd.DataFrames): DataFrames containing the requested data. The key for a given dataframe is that dataframe's Merger Tree Halo ID
    '''

    # Set up the data storage
    self.mtree_halos = {}

    # Get the halo filepaths
    ahf_filename = 'halo_*.dat'
    ahf_filepath_unexpanded = os.path.join( self.sdir, ahf_filename )
    halo_filepaths = glob.glob( ahf_filepath_unexpanded )

    # Loop over each file and load it
    for halo_filepath in halo_filepaths:

      # Load the data
      mtree_halo = pd.read_csv( halo_filepath, sep='\t', )

      # Delete a column that shows up as a result of formatting
      del mtree_halo[ 'Unnamed: 93' ]

      # Remove the annoying parenthesis at the end of each label.
      mtree_halo.columns = [ string.split( label, '(' )[0] for label in list( mtree_halo ) ]

      # Remove the pound sign in front of the first column's name
      mtree_halo = mtree_halo.rename( columns = {'#redshift':'redshift', ' ID':'ID'} )

      # Set the index, assuming we have 600 snapshots
      if index == 'snum':
        n_rows = mtree_halo.shape[0]
        mtree_halo['snum'] = range( 600, 600 - n_rows, -1)
        mtree_halo = mtree_halo.set_index( 'snum', )
      elif index == 'int':
        pass
      else:
        raise Exception( "index type not selected" )

      # Get a good key
      base_filename = string.split( halo_filepath, '/' )[-1]
      end_of_filename = base_filename[5:]
      halo_num = int( string.split( end_of_filename, '.' )[0] )

      # Store the data
      self.mtree_halos[ halo_num ] = mtree_halo

    # Make sure that at least the largest halo is traced for the full 600 snapshots.
    assert self.mtree_halos[0].shape[0] == 600

    return self.mtree_halos

  ########################################################################

  def get_halos( self, snum ):
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
    ahf_halos_path = self.get_filepath( snum, 'AHF_halos' )
    self.ahf_halos = pd.read_csv( ahf_halos_path, sep='\t', index_col=0 )

    # Delete a column that shows up as a result of formatting
    del self.ahf_halos[ 'Unnamed: 92' ]

    # Remove the annoying parenthesis at the end of each label.
    self.ahf_halos.columns = [ string.split( label, '(' )[0] for label in list( self.ahf_halos ) ]

    # Rename the index to a more suitable name, without the '#' and the (1)
    self.ahf_halos.index.names = ['ID']

    return self.ahf_halos

  ########################################################################

  def get_mtree_idx( self, snum ):
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
    ahf_mtree_idx_path = self.get_filepath( snum, 'AHF_mtree_idx' )
    self.ahf_mtree_idx = pd.read_csv( ahf_mtree_idx_path, delim_whitespace=True, names=['HaloID(1)', 'HaloID(2)'], skiprows=1  )

    return self.ahf_mtree_idx

  ########################################################################

  def get_filepath( self, snum, ahf_file_type ):
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
