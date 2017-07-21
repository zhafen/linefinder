#!/usr/bin/env python
'''Used to create an id file to track the IDs of.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import numpy as np

import galaxy_diver.analyze_data.particle_data as particle_data
import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################

class IDSelector( object ):

  def __init__( self, ptypes=None, snum_start=None, snum_end=None, snum_step=None, **kwargs ):
    '''
    Args:
      snum_start (int) : Starting snapshot number.
      snum_end (int) : Ending snapshot number.
      snum_step (int) : Snapshot step.

      ptypes (list of ints) : Types of particles to search through.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    # Make sure that all the arguments have been specified.
    for attr in vars( self ).keys():
      if attr == 'kwargs':
        continue
      if getattr( self, attr ) == None:
        raise Exception( '{} not specified'.format( attr ) )

    self.snums = range( snum_start, snum_end + 1, snum_step )

  ########################################################################

  @utilities.print_timer( 'Selecting all ids took' )
  def select_ids( self ):

    selected_ids = self.get_selected_ids()

    self.save_selected_ids()

  ########################################################################

  def get_selected_ids( self, data_filters ):
    '''Get a set of all ids that match a set of data filters.
    
    Args:
      data_filters (list of dicts): The data filters to apply.

    Returns:
      selected_ids (set): Set of selected ids.
    '''

    selected_ids = set()

    for snum in self.snums:
      for ptype in self.ptypes:

        kwargs = copy.deepcopy( self.kwargs )
        kwargs['snum'] = snum
        kwargs['ptype'] = ptype

        snapshot_id_selector = SnapshotIDSelector( **kwargs )
        selected_ids_snapshot = snapshot_id_selector.select_ids_snapshot( data_filters )

        selected_ids = selected_ids | selected_ids_snapshot

    return selected_ids

  ########################################################################

  def format_selected_ids( self, selected_ids ):
    '''Format the data back into arrays.

    Returns:
      ids (np.array): Array of selected IDs
      child_ids (np.array, optional) : Array of selected child IDS
    '''

    ids_arr = np.array( list( selected_ids ) )

    if len( ids_arr.shape ) > 1:
      ids, child_ids = ids_arr.transpose()
      return ids, child_ids

    else:
      return ids_arr

########################################################################
########################################################################

class SnapshotIDSelector( particle_data.ParticleData ):

  def __init__( self, **kwargs ):
    '''Class for selecting all the IDs that would go in a particular snapshot.
    Built on ParticleData.

    Keyword Args:
      All the usual requirements that would go into particle_data.ParticleData
    '''

    super( SnapshotIDSelector, self ).__init__( **kwargs )

  ########################################################################

  @utilities.print_timer( 'Selecting ids in a snapshot took' )
  def select_ids_snapshot( self, data_filters ):
    '''Select the IDs that match specified conditions in a given snapshot.
    
    Args:
      data_filters (list of dicts): The data filters to apply.
    '''

    self.filter_data( data_filters )

    selected_ids = self.get_ids()

    ids_set = self.format_ids( selected_ids )

    return ids_set

  ########################################################################

  def filter_data( self, data_filters ):
    '''Put the filters on the dataset
    
    Args:
      data_filters (list of dicts): The data filters to apply.
    '''

    for data_filter in data_filters:
      self.data_masker.mask_data( data_filter['data_key'], data_filter['data_min'], data_filter['data_max'] )

  ########################################################################

  def get_ids( self ):
    '''
    Returns:
      ids : IDs for particles that match the filtered requirements.
      child_ids (optional) : Child IDs for the particles that match the filtered requirements.
    '''

    ids = self.data_masker.get_masked_data( 'ID' )

    if not self.load_additional_ids:
      return ids

    else:
      child_ids = self.data_masker.get_masked_data( 'ChildID' )
      return ids, child_ids

  ########################################################################

  def format_ids( self, selected_ids ):
    '''Turns the ids into a set to be passed back.
    
    Args:
      selected_ids (np.array or list of np.arrays) : IDs to format into a set.
    
    Returns:
      ids_set (set) : IDs as a set.
    '''

    # When not loading IDs
    if not isinstance( selected_ids, tuple ):
      return set( selected_ids )

    else:
      return set( zip( *selected_ids ) )














