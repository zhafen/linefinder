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

class SnapshotIDSelector( object ):

  def __init__( self, **kwargs ):
    '''Select IDs satisfying certain conditions in a specified snapshot.
    Extra keyword arguments are passed to ParticleData()

    Keyword Args:
      sdir (str, required) : Directory containing the data.
      snum (int, required) : Snapshot number.
      ptype (int, required) : Particle type to search.
      load_additional_ids (bool, required) : Whether or not to also include the child IDs, if they are included.
      analysis_dir (str, required) : What directory the analysis data (e.g. AHF data) is stored in.
      ahf_index (int, required) : What index to use for the AHF data.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    self.p_data = particle_data.ParticleData( **kwargs )

  ########################################################################

  @utilities.print_timer( 'Selecting ids in the snapshot took' )
  def select_ids_snapshot( self, data_filters ):
    '''Select the IDs that match specified conditions in a given snapshot.
    
    Args:
      data_filters (list of dicts): The data filters to apply.
    '''

    self.filter_data()

    self.get_ids()

    self.format_ids()

  ########################################################################

  def filter_data( self, data_filters ):
    '''Put the filters on the dataset
    
    Args:
      data_filters (list of dicts): The data filters to apply.
    '''

    for data_filter in data_filters:
      self.p_data.data_masker.mask_data( data_filter['data_key'], data_filter['data_min'], data_filter['data_max'] )

















