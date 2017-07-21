#!/usr/bin/env python
'''Used to create an id file to track the IDs of.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import galaxy_diver.analyze_data.particle_data as particle_data
import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################

class IDSelector( object ):

  def __init__( self, ):

    pass

  ########################################################################

  @utilities.print_timer( 'Selecting all ids took' )
  def select_ids( self ):

    self.all_ids = set()
    for snum in snap_range:

      for ptype in ptypes:
        snapshot_id_selector = SnapshotIDSelector( **kwargs )
        selected_ids = snapshot_id_selector.select_ids_snapshot()

        self.all_ids = self.all_ids | selected_ids

    self.save_selected_ids()

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














