#!/usr/bin/env python
'''Means to associate particles with galaxies and halos at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import gc
import h5py
import numpy as np
import os
import scipy.spatial
import subprocess
import sys
import time

import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.utils.mp_utils as mp_utils
import galaxy_diver.utils.utilities as utilities

########################################################################

default = object()

########################################################################
########################################################################

class ParticleTrackGalaxyFinder( object ):
  '''Find the association with galaxies for entire particle tracks.'''

  @utilities.store_parameters
  def __init__( self,
    out_dir,
    tag,
    main_mt_halo_id,
    mtree_halos_index,
    ahf_data_dir = default,
    ptracks_tag = default,
    galaxy_cut = 0.5,
    length_scale = 'r_scale',
    ids_to_return = [ 'gal_id', 'mt_gal_id', 'd_sat', ],
    minimum_criteria = 'n_star',
    minimum_value = 10,
    n_processors = 1,
    ):
    '''Initialize.

    Args:
      out_dir (str) :
        Output directory, and directory the ptrack data is in.

      tag (str) :
        Identifying tag.

      main_mt_halo_id (int) :
        What is the ID of the main halo. To automatically choose via the most massive
        at z=0, set equal to None. HOWEVER, be warned that the most massive halo at z=0 *may not* be the main halo.

      mtree_halos_index (str or int) :
        The index argument to pass to AHFReader.get_mtree_halos().
        For most cases this should be the final snapshot number, but see AHFReader.get_mtree_halos's documentation.

      ahf_data_dir (str, optional) :
        Directory the AHF data is in. Defaults to the directory the simulation data is
        stored in, as found in the ptracks file.

      ptracks_tag (str, optional) :
        Identifying tag for the ptrack data. Defaults to tag.

      galaxy_cut (float, optional) :
        Anything within galaxy_cut*length_scale is counted as being inside the galaxy

      length_scale (str, optional) :
        Anything within galaxy_cut*length_scale is counted as being inside the galaxy.

      ids_to_return (list of strs, optional):
        The types of id you want to get out.

      minimum_criteria (str, optional) :
        Options...
        'n_star' -- halos must contain a minimum number of stars to count as containing a galaxy.
        'M_star' -- halos must contain a minimum stellar mass to count as containing a galaxy.

      minimum_value (int or float, optional) :
        The minimum amount of something (specified in minimum criteria)
        in order for a galaxy to count as hosting a halo.

      n_processors (int) :
        The number of processors to use. If parallel, expect significant memory usage.
    '''

    pass

  ########################################################################

  def find_galaxies_for_particle_tracks( self ):
    '''Main function.'''

    self.time_start = time.time()

    print "########################################################################"
    print "Starting Adding Galaxy and Halo IDs!"
    print "########################################################################"
    print "Using AHF data from this directory:\n    {}".format( self.ahf_data_dir )
    print "Data will be saved here:\n    {}".format( self.out_dir )
    sys.stdout.flush()

    self.read_data()

    if self.n_processors > 1:
      self.get_galaxy_identification_loop_parallel()
    else:
      self.get_galaxy_identification_loop()

    self.write_galaxy_identifications()

    time_end = time.time()

    print "########################################################################"
    print "Done Adding Galaxy and Halo IDs!"
    print "########################################################################"
    print "The data was saved at:\n    {}".format( self.save_filepath )
    print "Took {:.3g} seconds, or {:.3g} seconds per particle!".format( time_end - self.time_start,
                                                                      (time_end - self.time_start) / self.n_particles )

  ########################################################################

  def read_data( self ):
    '''Read the input data.

    Modifies:
      self.ptrack (h5py file) : Loaded tracked particle data.
      self.ahf_reader (AHFReader instance): For the ahf data.
    '''

    # Get the tag for particle tracking.
    if self.ptracks_tag is default:
      self.ptracks_tag = self.tag

    # Load the particle track data
    ptrack_filename = 'ptracks_{}.hdf5'.format( self.ptracks_tag )
    self.ptrack_filepath = os.path.join( self.out_dir, ptrack_filename )
    self.ptrack = h5py.File( self.ptrack_filepath, 'r' )

    if self.ahf_data_dir is default:
      self.ahf_data_dir = self.ptrack['parameters'].attrs['sdir']

    # Load the ahf data
    self.ahf_reader = read_ahf.AHFReader( self.ahf_data_dir )

  ########################################################################

  def get_galaxy_identification_loop( self ):
    '''Loop over all snapshots and identify the galaxy in each.

    Modifies:
      self.ptrack_gal_ids (dict) : Where the galaxy IDs are stored.
    '''

    # Loop over each included snapshot.
    n_snaps = self.ptrack['snum'][...].size
    for i in range( n_snaps ):

      # Get the particle positions
      particle_positions = self.ptrack['P'][...][ :, i ]
      
      # Get the data parameters to pass to GalaxyFinder
      kwargs = {
        'ahf_reader' : self.ahf_reader,
        'galaxy_cut' : self.galaxy_cut,
        'length_scale' : self.length_scale,
        'ids_to_return' : self.ids_to_return,
        'minimum_criteria' : self.minimum_criteria,
        'minimum_value' : self.minimum_value,

        'redshift' : self.ptrack['redshift'][...][ i ],
        'snum' : self.ptrack['snum'][...][ i ],
        'hubble' : self.ptrack.attrs['hubble'],
        'ahf_data_dir' : self.ahf_data_dir,
        'mtree_halos_index' : self.mtree_halos_index,
        'main_mt_halo_id' : self.main_mt_halo_id,
      }

      time_start = time.time()

      # Find the galaxy for a given snapshot
      galaxy_finder = GalaxyFinder( particle_positions, **kwargs )
      galaxy_and_halo_ids = galaxy_finder.find_ids()

      time_end = time.time()

      print 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'.format( kwargs['snum'],
                                                                                  kwargs['redshift'],
                                                                                  time_end - time_start )
      sys.stdout.flush()

      # Make the arrays to store the data in
      if not hasattr( self, 'ptrack_gal_ids' ):
        self.ptrack_gal_ids = {}
        for key in galaxy_and_halo_ids.keys():
          dtype = type( galaxy_and_halo_ids[key][0] )
          self.ptrack_gal_ids[key] = np.empty( ( galaxy_finder.n_particles, n_snaps ), dtype=dtype )

      # Store the data in the primary array
      for key in galaxy_and_halo_ids.keys():
        self.ptrack_gal_ids[key][ :, i ] = galaxy_and_halo_ids[key]

      # Try clearing up memory again, in case galaxy_finder is hanging around...
      del kwargs
      del galaxy_finder
      del galaxy_and_halo_ids
      gc.collect()

  ########################################################################

  def get_galaxy_identification_loop_parallel( self ):
    '''Loop over all snapshots and identify the galaxy in each.

    Modifies:
      self.ptrack_gal_ids (dict) : Where the galaxy IDs are stored.
    '''

    def get_galaxy_and_halo_ids( args ):
      '''Get the galaxy and halo ids for a single snapshot.'''

      particle_positions, kwargs = args
      
      time_start = time.time()

      # Find the galaxy for a given snapshot
      galaxy_finder = GalaxyFinder( particle_positions, **kwargs )
      galaxy_and_halo_ids = galaxy_finder.find_ids()

      time_end = time.time()

      print 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'.format( kwargs['snum'],
                                                                                  kwargs['redshift'],
                                                                                  time_end - time_start )
      sys.stdout.flush()

      # Try to avoid memory leaks
      del kwargs
      del galaxy_finder
      gc.collect()

      return galaxy_and_halo_ids

    n_snaps = self.ptrack['snum'][...].size
    n_particles = self.ptrack['P'][...].shape[0]

    # Loop over each included snapshot to get args
    all_args = []
    for i in range( n_snaps ):

      # Get the particle positions
      particle_positions = self.ptrack['P'][...][ :, i ]
      
      # Get the data parameters to pass to GalaxyFinder
      kwargs = {
        'ahf_reader' : None,
        'galaxy_cut' : self.galaxy_cut,
        'length_scale' : self.length_scale,
        'ids_to_return' : self.ids_to_return,
        'minimum_criteria' : self.minimum_criteria,
        'minimum_value' : self.minimum_value,

        'redshift' : self.ptrack['redshift'][...][ i ],
        'snum' : self.ptrack['snum'][...][ i ],
        'hubble' : self.ptrack.attrs['hubble'],
        'ahf_data_dir' : self.ahf_data_dir,
        'mtree_halos_index' : self.mtree_halos_index,
        'main_mt_halo_id' : self.main_mt_halo_id,
      }

      all_args.append( (particle_positions, kwargs) )

    # Actual parallel calculation
    galaxy_and_halo_ids_all = mp_utils.parmap( get_galaxy_and_halo_ids, all_args, self.n_processors )

    assert len( galaxy_and_halo_ids_all ) == n_snaps

    # Store the results
    for i, galaxy_and_halo_ids in enumerate( galaxy_and_halo_ids_all ):

      # Make the arrays to store the data in
      if not hasattr( self, 'ptrack_gal_ids' ):
        self.ptrack_gal_ids = {}
        for key in galaxy_and_halo_ids.keys():
          dtype = type( galaxy_and_halo_ids[key][0] )
          self.ptrack_gal_ids[key] = np.empty( ( n_particles, n_snaps ), dtype=dtype )

      # Store the data in the primary array
      for key in galaxy_and_halo_ids.keys():
        self.ptrack_gal_ids[key][ :, i ] = galaxy_and_halo_ids[key]

      # Try clearing up memory again, in case galaxy_finder is hanging around...
      del galaxy_and_halo_ids
      gc.collect()

  ########################################################################

  def write_galaxy_identifications( self ):
    '''Write the data, close the file, and print out information.'''

    # Get the number of particles, for use in reporting the time
    self.n_particles = self.ptrack[ 'Den' ][...].shape[0]

    # Close the old dataset
    self.ptrack.close()

    # Save the data.
    save_filename = 'galids_{}.hdf5'.format( self.tag )
    self.save_filepath = os.path.join( self.out_dir, save_filename )
    f = h5py.File( self.save_filepath )
    for key in self.ptrack_gal_ids.keys():
      f.create_dataset( key, data=self.ptrack_gal_ids[key] )

    # Store the main mt halo id (as identified by the larges value at the lowest redshift)
    if self.main_mt_halo_id is None:
      try:
        indice = self.ahf_reader.mtree_halos[0].index.max()
      except AttributeError:
        self.ahf_reader.get_mtree_halos( self.mtree_halos_index, 'smooth' )
        indice = self.ahf_reader.mtree_halos[0].index.max()
      m_vir_z0 = self.ahf_reader.get_mtree_halo_quantity( quantity='Mvir', indice=indice,
                                                          index=self.mtree_halos_index, tag='smooth' )
      f.attrs['main_mt_halo_id'] = m_vir_z0.argmax()
    else:
      f.attrs['main_mt_halo_id'] = self.main_mt_halo_id

    # Save the data parameters
    grp = f.create_group('parameters')
    for i, parameter in enumerate( self.stored_parameters ):
      grp.attrs[parameter] = getattr( self, parameter )

    # Save the current code version
    f.attrs['pathfinder_version'] = utilities.get_code_version( self )
    f.attrs['galaxy_diver_version'] = utilities.get_code_version( read_ahf, instance_type='module' )

    f.close()

########################################################################
########################################################################

class GalaxyFinder( object ):
  '''Find the association with galaxies and halos for a given set of particles at a given redshift.'''

  def __init__( self,
    particle_positions,
    ahf_reader = None,
    **kwargs ):
    '''Initialize.

    Args:
      particle_positions (np.array) :
        Positions with dimensions (n_particles, 3).

      ahf_reader (read_ahf object, optional) :
        An instance of an object that reads in the AHF data.
        If not given initiate one using the ahf_data_dir in kwargs

    Keyword Args:
      redshift (float, required) :
        Redshift the particles are at.

      snum (int, required) :
        Snapshot the particles correspond to.

      hubble (float, required) :
        Cosmological hubble parameter (little h)

      ahf_data_dir (str, required) :
        Directory the AHF data is in.

      mtree_halos_index (str or int, required)  :
        The index argument to pass to AHFReader.get_mtree_halos().
        For most cases this should be the final snapshot number, but see AHFReader.get_mtree_halos's documentation.

      The following will most likely be passed from ParticleTrackGalaxyFinder....

      galaxy_cut (float, required) :
        The fraction of the length scale a particle must be inside to be counted as part
        of a galaxy.

      length_scale (str, required) :
        Anything within galaxy_cut*length_scale is counted as being inside the galaxy.

      ids_to_return (list of strs, required) :
        The types of id you want to get out.

      minimum_stellar_mass (float, required if no minimum_num_stars) :
        The minimum stellar mass a halo must contain to
        count as containing a galaxy.

      minimum_num_stars (int, required if no minimum_stellar_mass) :
        The minimum number of stars a halo must contain to count as containing a galaxy.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    # Setup the default ahf_reader
    if ahf_reader is None:
      self.ahf_reader = read_ahf.AHFReader( self.kwargs['ahf_data_dir'] )

    # In the case of a minimum stellar mass, we need to divide it by 1/h when getting its values out.
    if self.kwargs['minimum_criteria'] == 'M_star':
      self.min_conversion_factor = self.kwargs['hubble'] 
    else:
      self.min_conversion_factor = 1

    # Derived properties
    self.n_particles = self.particle_positions.shape[0]

  ########################################################################

  def find_ids( self ):
    '''Find relevant halo and galaxy IDs.

    Returns:
      galaxy_and_halo_ids (dict): Keys are...
      Parameters:
        halo_id (np.array of ints): ID of the least-massive halo the particle is part of.
        host_halo_id (np.array of ints): ID of the host halo the particle is part of.
        gal_id (np.array of ints): ID of the smallest galaxy the particle is part of.
        host_gal_id (np.array of ints): ID of the host galaxy the particle is part of.
        mt_halo_id (np.array of ints): Merger tree ID of the least-massive halo the particle is part of.
        mt_gal_id (np.array of ints): Merger tree ID of the smallest galaxy the particle is part of.
    '''

    # Dictionary to store the data in.
    galaxy_and_halo_ids = {}

    try:
      # Load the ahf data
      self.ahf_reader.get_halos( self.kwargs['snum'] )

    # Typically halo files aren't created for the first snapshot.
    # Account for this.
    except NameError:
      if self.kwargs['snum'] == 0:
        for id_type in self.kwargs['ids_to_return']:
          galaxy_and_halo_ids[id_type] = np.empty( self.n_particles )
          galaxy_and_halo_ids[id_type].fill( -2. )

        return galaxy_and_halo_ids

      else:
        raise KeyError( 'AHF data not found for snum {} in {}'.format( self.kwargs['snum'],
                                                                       self.kwargs['ahf_data_dir'] ) )
    
    # Actually get the data
    for id_type in self.kwargs['ids_to_return']:
      if id_type == 'halo_id':
        galaxy_and_halo_ids['halo_id'] = self.find_halo_id()
      elif id_type == 'host_halo_id':
        galaxy_and_halo_ids['host_halo_id'] = self.find_host_id()
      elif id_type == 'gal_id':
        galaxy_and_halo_ids['gal_id'] = self.find_halo_id( self.kwargs['galaxy_cut'] )
      elif id_type == 'host_gal_id':
        galaxy_and_halo_ids['host_gal_id'] = self.find_host_id( self.kwargs['galaxy_cut'] )
      elif id_type == 'mt_halo_id':
        galaxy_and_halo_ids['mt_halo_id'] = self.find_halo_id( type_of_halo_id='mt_halo_id' )
      elif id_type == 'mt_gal_id':
        galaxy_and_halo_ids['mt_gal_id'] = self.find_halo_id( self.kwargs['galaxy_cut'], type_of_halo_id='mt_halo_id' )
      elif id_type == 'd_gal':
        galaxy_and_halo_ids['d_gal'] = self.find_d_gal()
      elif id_type == 'd_sat':
        galaxy_and_halo_ids['d_sat'] = self.find_d_sat()
      else:
        raise Exception( "Unrecognized id_type" )
    
    return galaxy_and_halo_ids

  ########################################################################

  def find_d_gal( self ):
    '''Find the distance to the center of the closest halo that contains a galaxy.

    Returns:
      d_gal (np.ndarray) : For particle i, d_gal[i] is the distance in pkpc to the center of the nearest galaxy.
    '''
  
    return np.min( self.dist_to_all_valid_halos, axis=1 )

  ########################################################################

  def find_d_sat( self ):
    '''Find the distance to the center of the closest halo that contains a satellite galaxy.

    Returns:
      d_sat (np.ndarray) :
        For particle i, d_sat[i] is the distance in pkpc to the center of the nearest galaxy, besides the main galaxy.
    '''

    self.ahf_reader.get_mtree_halos( self.kwargs['mtree_halos_index'], 'smooth' )

    # The indice for the main galaxy is the same as the AHF_halos ID for it.
    mtree_halo = self.ahf_reader.mtree_halos[ self.kwargs['main_mt_halo_id'] ]
    if self.kwargs['snum'] < mtree_halo.index.min():
      ind_main_gal_in_valid_inds = np.array( [] )
    else:
      ind_main_gal = mtree_halo['ID'][ self.kwargs['snum'] ]

      valid_halo_ind_is_main_gal_ind = self.valid_halo_inds == ind_main_gal 
      ind_main_gal_in_valid_inds = np.where( valid_halo_ind_is_main_gal_ind )[0]

    if ind_main_gal_in_valid_inds.size == 0:
      return np.min( self.dist_to_all_valid_halos, axis=1 )

    elif ind_main_gal_in_valid_inds.size == 1:
      dist_to_all_valid_sats = np.delete( self.dist_to_all_valid_halos, ind_main_gal_in_valid_inds[0], axis=1 )
      return np.min( dist_to_all_valid_sats, axis=1 )

    else:
      raise Exception( "valid_ind_main_gal too big, is size {}".format( valid_ind_main_gal.size ) )

  ########################################################################

  def find_host_id( self, radial_cut_fraction=1. ):
    '''Find the host halos our particles are inside of some radial cut of.
    This is the host ID at a given snapshot, and is not the same as the merger tree halo ID.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

    Returns:
      host_halo (np.array of ints): Shape ( n_particles, ). 
        The ID of the least massive substructure the particle's part of.
        If it's -1, then the halo ID is the host ID.
        If it's -2, then that particle is not part of any halo, within radial_cut_fraction*length_scale .
    '''

    # Get the halo ID
    halo_id = self.find_halo_id( radial_cut_fraction )

    ahf_host_id =  self.ahf_reader.ahf_halos['hostHalo']

    # Handle the case where we have an empty ahf_halos, because there are no halos at that redshift.
    # In this case, the ID will be -2 throughout
    if ahf_host_id.size == 0:
      return halo_id

    # Get the host halo ID
    host_id = ahf_host_id[ halo_id ]

    # Fix the invalid values (which come from not being associated with any halo)
    host_id_fixed = np.ma.fix_invalid( host_id, fill_value=-2 )

    return host_id_fixed.data.astype( int )

  ########################################################################

  def find_halo_id( self, radial_cut_fraction=1., type_of_halo_id='halo_id' ):
    '''Find the smallest halos our particles are inside of some radial cut of (we define this as the halo ID).
    In the case of using MT halo ID, we actually find the most massive our particles are inside some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.
      type_of_halo_id (str): If 'halo_id' then this is the halo_id at a given snapshot.
                             If 'mt_halo_id' then this is the halo_id according to the merger tree.

    Returns:
      halo_id (np.array of ints): Shape ( n_particles, ). 
        The ID of the least massive substructure the particle's part of.
        In the case of using the 'mt_halo_id', this is the ID of the most massive merger tree halo the particle's part of.
        If it's -2, then that particle is not part of any halo, within radial_cut_fraction*length_scale .
    '''

    # Choose parameters of the rest of the function based on what type of halo ID we're using
    if type_of_halo_id == 'halo_id':

      # Get the virial masses. It's okay to leave in comoving, since we're just finding the minimum
      m_vir = self.ahf_reader.ahf_halos['Mvir']

      # Handle the case where we have an empty ahf_halos, because there are no halos at that redshift.
      # In this case, the halo ID will be -2 throughout
      if m_vir.size == 0:
        halo_id = np.empty( self.n_particles )
        halo_id.fill( -2. )
        return halo_id

      # Functions that change.
      find_containing_halos_fn = self.find_containing_halos
      arg_extremum_fn = np.argmin
      extremum_fn = np.min

    elif type_of_halo_id == 'mt_halo_id':

      # Functions that change.
      find_containing_halos_fn = self.find_mt_containing_halos
      arg_extremum_fn = np.argmax
      extremum_fn = np.max

      # Get the virial masses. It's okay to leave in comoving, since we're just finding the maximum
      m_vir = self.ahf_reader.get_mtree_halo_quantity( quantity='Mvir', indice=self.kwargs['snum'],
                                                       index=self.kwargs['mtree_halos_index'], tag='smooth' )

    else:
      raise Exception( "Unrecognized type_of_halo_id" )

    # Get the cut
    part_of_halo = find_containing_halos_fn( radial_cut_fraction=radial_cut_fraction )

    # Mask the data
    tiled_m_vir = np.tile( m_vir, ( self.n_particles, 1 ) )
    tiled_m_vir_ma = np.ma.masked_array( tiled_m_vir, mask=np.invert( part_of_halo ), )

    # Take the extremum of the masked data
    if type_of_halo_id == 'halo_id':
      halo_id = arg_extremum_fn( tiled_m_vir_ma, axis=1 )
    elif type_of_halo_id == 'mt_halo_id':
      halo_ind = arg_extremum_fn( tiled_m_vir_ma, axis=1 )
      halo_ids = np.array( sorted( self.ahf_reader.mtree_halos.keys() ) )
      halo_id = halo_ids[halo_ind]
    
    # Account for the fact that the argmin defaults to 0 when there's nothing there
    mask = extremum_fn( tiled_m_vir_ma, axis=1 ).mask
    halo_id = np.ma.filled( np.ma.masked_array(halo_id, mask=mask), fill_value=-2 )

    return halo_id

  ########################################################################

  def find_containing_halos( self, radial_cut_fraction=1. ):
    '''Find which halos our particles are inside of some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

    Returns:
      part_of_halo (np.array of bools): Shape (n_particles, n_halos). 
        If index [i, j] is True, then particle i is inside radial_cut_fraction*length_scale of the jth halo.
    '''

    # Get the radial cut
    radial_cut = radial_cut_fraction*self.ahf_halos_length_scale_pkpc[self.valid_halo_inds]

    # Find the halos that our particles are part of (provided they passed the minimum cut)
    part_of_halo_success = self.dist_to_all_valid_halos < radial_cut[np.newaxis,:]

    # Get the full array out
    part_of_halo = np.zeros( (self.n_particles, self.ahf_halos_length_scale_pkpc.size) ).astype( bool )
    part_of_halo[:,self.valid_halo_inds] = part_of_halo_success

    return part_of_halo

  ########################################################################

  @property
  def ahf_halos_length_scale_pkpc( self ):

    if not hasattr( self, '_ahf_halos_length_scale_pkpc' ):

      # Get the relevant length scale
      if self.kwargs['length_scale'] == 'R_vir':
        length_scale = self.ahf_reader.ahf_halos['Rvir']
      elif self.kwargs['length_scale'] == 'r_scale':
        # Get the files containing the concentration (counts on it being already calculated beforehand)
        self.ahf_reader.get_halos_add( self.ahf_reader.ahf_halos_snum )

        # Get the scale radius
        r_vir = self.ahf_reader.ahf_halos['Rvir']
        length_scale = r_vir/self.ahf_reader.ahf_halos_add['cAnalytic']
      else:
        raise KeyError( "Unspecified length scale" )
      self._ahf_halos_length_scale_pkpc = length_scale/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']

    return self._ahf_halos_length_scale_pkpc

  ########################################################################

  @property
  def dist_to_all_valid_halos( self ):
    '''
    Returns:
      dist_to_all_valid_halos (np.ndarray) :
        Distance between the particle positions and all *.AHF_halos halos containing a galaxy.
    '''

    if not hasattr( self, '_dist_to_all_valid_halos' ):
        
      # Get the halo positions
      halo_pos_comov = np.array([
        self.ahf_reader.ahf_halos['Xc'],
        self.ahf_reader.ahf_halos['Yc'],
        self.ahf_reader.ahf_halos['Zc'],
      ]).transpose()
      halo_pos = halo_pos_comov/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']
      halo_pos_selected = halo_pos[self.valid_halo_inds]

      # Get the distances
      # Output is ordered such that dist[:,0] is the distance to the center of halo 0 for each particle
      self._dist_to_all_valid_halos = scipy.spatial.distance.cdist( self.particle_positions, halo_pos_selected )

    return self._dist_to_all_valid_halos

  ########################################################################

  @property
  def valid_halo_inds( self ):
    '''
    Returns:
      valid_halo_inds (np.ndarray) :
        Indices of *AHF_halos halos that satisfy our minimum criteria for containing a galaxy.
    '''

    if not hasattr( self, '_valid_halo_inds' ):

      # Apply a cut on containing a minimum amount of stars
      min_criteria = self.ahf_reader.ahf_halos[ self.kwargs['minimum_criteria'] ]
      has_minimum_value = min_criteria/self.min_conversion_factor >= self.kwargs['minimum_value']

      # Figure out which indices satisfy the criteria and choose only those halos
      self._valid_halo_inds = np.where( has_minimum_value )[0]

    return self._valid_halo_inds

  ########################################################################

  def find_mt_containing_halos( self, radial_cut_fraction=1. ):
    '''Find which MergerTrace halos our particles are inside of some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

    Returns:
      part_of_halo (np.array of bools): Shape (n_particles, n_halos). 
        If index [i, j] is True, then particle i is inside radial_cut_fraction*length_scale of the jth halo, defined
          via the MergerTrace ID.
    '''

    # Load up the merger tree data
    self.ahf_reader.get_mtree_halos( self.kwargs['mtree_halos_index'], 'smooth' )

    part_of_halo = []
    for halo_id in self.ahf_reader.mtree_halos.keys():
      mtree_halo = self.ahf_reader.mtree_halos[ halo_id ]

      # Only try to get the data if we're in the range we actually have the halos for.
      above_minimum_snap = self.kwargs['snum'] >= mtree_halo.index.min()

      # Only try to get the data if we have the minimum stellar mass
      if above_minimum_snap:
        halo_value = mtree_halo[ self.kwargs['minimum_criteria'] ][ self.kwargs['snum'] ]/self.min_conversion_factor 
        has_minimum_value = halo_value >= self.kwargs['minimum_value']
      else:
        # If it's not at the point where it can be traced, it definitely doesn't have the minimum stellar mass.
        has_minimum_value = False

      # Usual case
      if has_minimum_value:

        # Get the halo position
        halo_pos_comov = np.array([
          mtree_halo['Xc'][ self.kwargs['snum'] ],
          mtree_halo['Yc'][ self.kwargs['snum'] ],
          mtree_halo['Zc'][ self.kwargs['snum'] ],
        ])
        halo_pos = halo_pos_comov/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']

        # Make halo_pos 2D for compatibility with cdist
        halo_pos = halo_pos[np.newaxis]

        # Get the distances
        dist = scipy.spatial.distance.cdist( self.particle_positions, halo_pos )

        # Get the relevant length scale
        if self.kwargs['length_scale'] == 'R_vir':
          length_scale = mtree_halo['Rvir'][ self.kwargs['snum'] ]
        elif self.kwargs['length_scale'] == 'r_scale':
          # Get the scale radius
          r_vir = mtree_halo['Rvir'][ self.kwargs['snum'] ]
          length_scale = r_vir/mtree_halo['cAnalytic'][ self.kwargs['snum'] ]
        else:
          raise KeyError( "Unspecified length scale" )
        length_scale_pkpc = length_scale/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']

        # Get the radial distance
        radial_cut = radial_cut_fraction*length_scale_pkpc
        
        # Find if our particles are part of this halo
        part_of_this_halo = dist < radial_cut

      # Case where there isn't a main halo at that redshift.
      else:
        part_of_this_halo = np.zeros( (self.n_particles, 1) ).astype( bool )

      part_of_halo.append( part_of_this_halo )

    # Format part_of_halo correctly
    part_of_halo = np.array( part_of_halo ).transpose()[0]

    return part_of_halo
