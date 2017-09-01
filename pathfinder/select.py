#!/usr/bin/env python
'''Used to create an id file to track the IDs of.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import gc
import h5py
import numpy as np
import os
import shutil
import sys
import time

import galaxy_diver.analyze_data.particle_data as particle_data
import galaxy_diver.utils.mp_utils as mp_utils
import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################

class IDSelector( object ):

  @utilities.store_parameters
  def __init__( self,
    out_dir,
    tag,
    snum_start,
    snum_end,
    snum_step,
    p_types,
    snapshot_kwargs,
    n_processors=1,
    ):
    '''
    Args:
      out_dir (str) :
        Where to store the data.

      tag (str) :
        Tag to give the filename.

      snum_start (int) :
        Starting snapshot number.

      snum_end (int) :
        Ending snapshot number.

      snum_step (int) :
        Snapshot step.

      p_types (list of ints) :
        Types of particles to search through.

      snapshot_kwargs (dict) :
        Arguments to pass to SnapshotIDSelector. Can be the full range of arguments passed to
        particle_data.ParticleData

      n_processors (int, optional) :
        The number of processers to run the ID selector with. Parallelizes by opening multiple snapshots
        at once (that's the most time-consumptive part of the code), so requires a large memory node, most likely.
    '''

    self.snums = range( self.snum_start, self.snum_end + 1, self.snum_step )

  ########################################################################

  @utilities.print_timer( 'Selecting all ids took' )
  def select_ids( self, data_filters ):
    '''Save a set of all ids that match a set of data filters to a file.
    
    Args:
      data_filters (list of dicts): The data filters to apply.
    '''

    print( "########################################################################" )
    print( "Selecting IDs" )
    print( "########################################################################" )
    sys.stdout.flush()

    if self.n_processors > 1:
      selected_ids = self.get_selected_ids_parallel( data_filters )
    else:
      selected_ids = self.get_selected_ids( data_filters )

    selected_ids_formatted = self.format_selected_ids( selected_ids )

    self.save_selected_ids( selected_ids_formatted )

    print( "########################################################################" )
    print( "Done!" )
    sys.stdout.flush()

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
      for ptype in self.p_types:

        kwargs = dict( self.snapshot_kwargs )
        kwargs['snum'] = snum
        kwargs['ptype'] = ptype

        time_start = time.time()

        snapshot_id_selector = SnapshotIDSelector( **kwargs )
        selected_ids_snapshot = snapshot_id_selector.select_ids_snapshot( data_filters )

        time_end = time.time()

        print( "Ptype {}, Snapshot {}, took {:.3g} seconds".format( ptype, snum, time_end - time_start ) )
        sys.stdout.flush()

        selected_ids = selected_ids | selected_ids_snapshot

        # Vain effort to free memory (that, stunningly, actually works!!)
        del kwargs
        del snapshot_id_selector
        del selected_ids_snapshot
        gc.collect()

    return selected_ids

  ########################################################################

  def get_selected_ids_parallel( self, data_filters ):
    '''Parallel version of self.get_selected_ids(). Requires a lot of memory, because it will have multiple 
    snapshots open at once.
    
    Args:
      data_filters (list of dicts): The data filters to apply.

    Returns:
      selected_ids (set): Set of selected ids.
    '''

    selected_ids = set()

    args = []
    for snum in self.snums:
      for ptype in self.p_types:

        kwargs = dict( self.snapshot_kwargs )
        kwargs['snum'] = snum
        kwargs['ptype'] = ptype

        args.append( ( data_filters, kwargs ) )

    results = mp_utils.parmap( self.get_selected_ids_snapshot, args, self.n_processors, set_case=True )

    selected_ids = set.union( *results )

    return selected_ids

  def get_selected_ids_snapshot( self, args ):
    '''Get the IDs for a particular snapshot. Formatted this way primarily for parallelization.

    Args:
      args (data_filters, kwargs) : Information needed to get the IDs out for a snapshot.

    Returns:
      selected_ids_snapshot (set) : The IDs in a snapshot that fit the required condition.
    '''

    data_filters, kwargs = args

    time_start = time.time()

    snapshot_id_selector = SnapshotIDSelector( **kwargs )
    selected_ids_snapshot = snapshot_id_selector.select_ids_snapshot( data_filters )

    time_end = time.time()

    print( "Ptype {}, Snapshot {}, took {:.3g} seconds".format( kwargs['ptype'], kwargs['snum'],
                                                                time_end - time_start ) )

    # Vain effort to free memory (that, stunningly, actually works!!)
    # Though I haven't checked if it works in multiprocessing
    del kwargs
    del snapshot_id_selector
    gc.collect()

    return selected_ids_snapshot

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

  def save_selected_ids( self, selected_ids_formatted ):

    # Open up the file to save the data in.
    ids_filename =  'ids_full_{}.hdf5'.format( self.tag )
    self.ids_filepath = os.path.join( self.out_dir, ids_filename )
    f = h5py.File( self.ids_filepath, 'a' )

    # Save the data
    if isinstance( selected_ids_formatted, tuple ):
      ids, child_ids = selected_ids_formatted
      f.create_dataset( 'target_ids', data=ids )
      f.create_dataset( 'target_child_ids', data=child_ids )

    else:
      ids = selected_ids_formatted
      f.create_dataset( 'target_ids', data=ids )

    # Create groups for the parameters
    grp = f.create_group('parameters')
    subgrp = f.create_group('parameters/snapshot_parameters')

    # Save the data parameters
    for parameter in self.stored_parameters:
      if parameter != 'snapshot_kwargs':
        grp.attrs[parameter] = getattr( self, parameter )

    # Save the snapshot parameters too
    for key in self.snapshot_kwargs.keys():
      subgrp.attrs[key] = self.snapshot_kwargs[key]

    # Save how many processors we used.
    grp.attrs['n_processors'] = self.n_processors

    # Save the current code versions
    f.attrs['pathfinder_version'] = utilities.get_code_version( self )
    f.attrs['galaxy_diver_version'] = utilities.get_code_version( particle_data, instance_type='module' )

    f.close()

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

########################################################################
########################################################################

class IDSampler( object ):

  def __init__( self, sdir, tag, n_samples=100000, ignore_child_particles=False, ignore_duplicates=False  ):
    '''Sample an ID file to obtain and save a subset of size n_samples.
    Assumes the full set of IDs are saved as ids_full_tag.hdf5, and will save the sampled IDs as ids_tag.hdf5.

    Args:
      sdir (str, required) : Directory the IDs are in, as well as the directory the sampled IDs will be saved in.
      tag (str, required) : Identifying string for the ID file.
      n_samples (int, optional) : Number of samples to sample.
      ignore_child_particles (bool, optional) : Whether or not to ignore particles with non-zero child ID when sampling
      ignore_duplicates (bool, optional) : Whether or not to ignore particles that have duplicate IDs at the final
        snapshot.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

  ########################################################################

  @utilities.print_timer( 'Sampling ids took' )
  def sample_ids( self ):
    '''Sample a saved ID file for a subset, which can then be run through particle tracking.
    '''

    print( "########################################################################" )
    print( "Sampling IDs" )
    print( "########################################################################" )
    sys.stdout.flush()

    print( "Copying and opening full ids..." )
    sys.stdout.flush()
    self.copy_and_open_full_ids()

    print( "Choosing particles to sample..." )
    sys.stdout.flush()
    self.choose_particles_to_sample()

    print( "Choosing sample indices..." )
    sys.stdout.flush()
    self.choose_sample_inds()

    print( "Saving data..." )
    sys.stdout.flush()
    self.save_sampled_ids()

    print( "########################################################################" )
    print( "Done!" )
    sys.stdout.flush()

  ########################################################################

  def copy_and_open_full_ids( self ):
    '''Copy the full id file and save them.

    Modifies:
      self.f (h5py file) : Opens and creates the file.
    '''

    full_id_filename = 'ids_full_{}.hdf5'.format( self.tag )
    id_filename = 'ids_{}.hdf5'.format( self.tag )

    full_id_filepath = os.path.join( self.sdir, full_id_filename )
    id_filepath = os.path.join( self.sdir, id_filename )
  
    shutil.copyfile( full_id_filepath, id_filepath )

    self.f = h5py.File( id_filepath, 'a' )

  ########################################################################

  def identify_child_particles( self ):
    '''Get the IDs and Child IDs of all particles that have non-zero Child IDs.

    Returns:
      child_particles (set) : A set of all the IDs and Child IDs that have non-zero Child IDs.
    '''

    split = np.where( self.f['target_child_ids'][...] != 0 )[0]

    split_ids = self.f['target_ids'][...][split]
    split_child_ids = self.f['target_child_ids'][...][split]

    return utilities.arrays_to_set( split_ids, split_child_ids )

  ########################################################################

  def identify_duplicate_ids( self ):
    '''Get all IDs that have duplicates at the latest snapshot. This draws data from the same sample as the original.
    
    Returns:
      duplicate_ids (set) : A set of all IDs that have duplicates at snum_end.
    '''

    assert 'target_child_ids' not in self.f.keys(), "Identifying duplicate IDs does not work with new ID system."

    duplicate_ids = set()
    id_sets = []
    for ptype in self.f['parameters'].attrs['p_types']:

      p_data_kwargs = dict( self.f['parameters/snapshot_parameters'].attrs )
      p_data_kwargs['snum'] = self.f['parameters'].attrs['snum_end']
      p_data_kwargs['ptype'] = ptype
      p_data = particle_data.ParticleData( **p_data_kwargs )

      duplicate_ids = duplicate_ids | set( p_data.find_duplicate_ids() )

      # Get the IDs out, so that we can find duplicates where the particles are of different types
      id_sets.append( set( p_data.get_data( 'ID' ) ) )

    duplicate_ids = duplicate_ids | set.intersection( *id_sets )

    return duplicate_ids

  ########################################################################

  def choose_particles_to_sample( self ):
    '''Choose which particles to subsample. This is done by subtracting out any IDs that we don't want to use for some
    reason (e.g. if we're ignoring IDs that have duplicates).

    Modifies:
      self.ids_to_sample (np.ndarray) : An array of the IDs to sample.
      self.child_ids_to_sample (np.ndarray, optional) : An array of the child IDs to sample.
    '''

    if self.ignore_duplicates or self.ignore_child_particles:

      # Get the IDs in the right format.
      if 'target_child_ids' in self.f.keys():
        ids_set = utilities.arrays_to_set( self.f['target_ids'][...], self.f['target_child_ids'][...] )
      else:
        ids_set =  set( self.f['target_ids'][...] )

      if self.ignore_duplicates:
        print( "  Removing duplicate particles..." )
        ids_set -= self.identify_duplicate_ids()

      if self.ignore_child_particles:
        print( "  Removing particles with non-zero child IDs..." )
        ids_set -= self.identify_child_particles()

      assert not ( self.ignore_child_particles and self.ignore_duplicates ), \
        "Code currently cannot handle both ignoring child particles and ignoring duplicates."

      # Store the data again
      if 'target_child_ids' in self.f.keys():
        self.ids_to_sample, self.child_ids_to_sample = utilities.set_to_arrays( ids_set )
      else:
        self.ids_to_sample = np.array( list( ids_set ) )

    else:

      self.ids_to_sample = self.f['target_ids'][...]
      if 'target_child_ids' in self.f.keys():
        self.child_ids_to_sample = self.f['target_child_ids'][...]

  ########################################################################

  def choose_sample_inds( self ):
    '''Select the indices of the target IDs to sample.

    Modifies:
      self.sample_inds (np.array of ints) : Indices of the target IDs to sample.
    '''

    inds = np.array( range( self.ids_to_sample.size ) )

    self.sample_inds = np.random.choice( inds, self.n_samples, replace=False )

  ########################################################################

  def save_sampled_ids( self ):
    '''Save the IDs, now that we have the indices of the IDs we want sampled.

    Modifies:
      self.f (h5py file) : Replaces target_ids and target_child_ids with sampled versions.
    '''

    # Save the total number of particles in the original
    self.f.attrs['n_particles'] = self.f['target_ids'].size

    target_ids_to_save = self.ids_to_sample[self.sample_inds]
    del self.f['target_ids']
    self.f['target_ids'] = target_ids_to_save

    if 'target_child_ids' in self.f.keys():
      target_child_ids_to_save = self.child_ids_to_sample[self.sample_inds]
      del self.f['target_child_ids']
      self.f['target_child_ids'] = target_child_ids_to_save

    self.f['parameters'].attrs['n_samples'] = self.n_samples
    self.f['parameters'].attrs['ignore_child_particles'] = self.ignore_child_particles
    self.f['parameters'].attrs['ignore_duplicates'] = self.ignore_duplicates
    self.f['parameters'].attrs['sampled_from_full_id_list'] = True

    self.f.close()












