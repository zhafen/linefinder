#!/usr/bin/env python
'''Means to associate particles with galaxies and halos at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os
import scipy.spatial
import time

import galaxy_diver.read_ahf

########################################################################
########################################################################

class ParticleTrackGalaxyFinder( object ):
  '''Find the association with galaxies for entire particle tracks.'''

  def __init__( self, galaxy_cut=0.1, ids_to_return=[ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id', 'mt_halo_id', 'mt_gal_id' ], **kwargs ):
    '''Initialize.

    Args:
      galaxy_cut (float, optional): Anything within galaxy_cut*R_vir is counted as being inside the virial radius.
      ids_to_return (list of strs, optional): The types of id you want to get out.
      mtree_halos_index: The index argument to pass to AHFReader.get_mtree_halos(). For most cases this should be the final
                        snapshot number, but see AHFReader.get_mtree_halos's documentation.

    Keyword Args:
      sdir (str): Directory the AHF data is in.
      tracking_dir (str): Directory the ptrack data is in.
      tag (str): Identifying tag for the ptrack data
    '''

    self.kwargs = kwargs

    self.galaxy_cut = galaxy_cut
    self.ids_to_return = ids_to_return

  ########################################################################

  def find_galaxies_for_particle_tracks( self ):
    '''Main function.'''

    self.start_data_processing()

    # Loop over each included snapshot.
    n_snaps = self.ptrack['snum'][...].size
    for i in range( n_snaps ):

      # Get the particle positions
      particle_positions = self.ptrack['p'][...][ :, i ]
      
      # Get the data parameters to pass to GalaxyFinder
      kwargs = {
        'redshift' : self.ptrack['redshift'][...][ i ],
        'snum' : self.ptrack['snum'][...][ i ],
        'hubble' : self.ptrack.attrs['hubble'],
        'sdir' : self.kwargs['sdir'],
        'mtree_halos_index' : self.kwargs['mtree_halos_index'],
      }

      print 'Snapshot {:>3} | redshift {:>7.3g}'.format( kwargs['snum'], kwargs['redshift'], )

      # Find the galaxy for a given snapshot
      galaxy_finder = GalaxyFinder( particle_positions, ahf_reader=self.ahf_reader, **kwargs )
      galaxy_and_halo_ids = galaxy_finder.find_ids( ids_to_return=self.ids_to_return, galaxy_cut=self.galaxy_cut )

      # Make the arrays to store the data in
      if not hasattr( self, 'ptrack_gal_ids' ):
        self.ptrack_gal_ids = {}
        for key in galaxy_and_halo_ids.keys():
          self.ptrack_gal_ids[key] = np.empty( ( galaxy_finder.n_particles, n_snaps ), dtype=int )

      # Store the data in the primary array
      for key in galaxy_and_halo_ids.keys():
        self.ptrack_gal_ids[key][ :, i ] = galaxy_and_halo_ids[key]

    self.finish_data_processing()

  ########################################################################

  def start_data_processing( self ):
    '''Open up the ptrack file and print out information.'''

    self.time_start = time.time()

    print "########################################################################"
    print "Starting Adding Galaxy and Halo IDs!"
    print "########################################################################"
    print "Using AHF data from this directory:\n    {}".format( self.kwargs['sdir'] )
    print "Data will be saved here:\n    {}".format( self.kwargs['tracking_dir'] )

    # Load the particle track data
    ptrack_filename = 'ptrack_{}.hdf5'.format( self.kwargs['tag'] )
    self.ptrack_filepath = os.path.join( self.kwargs['tracking_dir'], ptrack_filename )
    self.ptrack = h5py.File( self.ptrack_filepath, 'a' )

    # Load the ahf data
    self.ahf_reader = galaxy_diver.read_ahf.AHFReader( self.kwargs['sdir'] )

  ########################################################################

  def finish_data_processing( self ):
    '''Write the data, close the file, and print out information.'''

    # Get the number of particles, for use in reporting the time
    n_particles = self.ptrack[ 'rho' ][...].shape[0]

    # Close the old dataset
    self.ptrack.close()

    # Save the data.
    # Load the particle track data
    save_filename = 'galfind_{}.hdf5'.format( self.kwargs['tag'] )
    save_filepath = os.path.join( self.kwargs['tracking_dir'], save_filename )
    f = h5py.File( save_filepath )
    for key in self.ptrack_gal_ids.keys():
      f.create_dataset( key, data=self.ptrack_gal_ids[key] )

    # Store the main mt halo id
    m_vir_z0 = self.ahf_reader.get_mtree_halo_quantity( quantity='Mvir', indice=600, index=self.kwargs['mtree_halos_index'], tag='smooth' )
    f.attrs['main_mt_halo_id'] = m_vir_z0.argmax()

    # Save the data parameters
    for key in self.kwargs.keys():
      f.attrs[key] = self.kwargs[key]

    f.close()

    time_end = time.time()
    print "########################################################################"
    print "Done Adding Galaxy and Halo IDs!"
    print "########################################################################"
    print "The following particle track file was updated:\n    {}".format( self.ptrack_filepath )
    print "Took {:.3g} seconds, or {:.3g} seconds per particle!".format( time_end - self.time_start, (time_end - self.time_start) / n_particles )

########################################################################
########################################################################

class GalaxyFinder( object ):
  '''Find the association with galaxies and halos for a given set of particles at a given redshift.'''

  def __init__( self, particle_positions, ahf_reader=None, **kwargs ):
    '''Initialize.

    Args:
      particle_positions (np.array): Positions with dimensions (n_particles, 3).
      ahf_reader (galaxy_diver.read_ahf object, optional): An instance of an object that reads in the AHF data. If not given initiate one using the sdir in kwargs

    Keyword Args:
      redshift (float): Redshift the particles are at.
      snum (int): Snapshot the particles correspond to.
      hubble (float): Cosmological hubble parameter (little h)
      sdir (str): Directory the AHF data is in.
    '''

    self.kwargs = kwargs

    self.particle_positions = particle_positions

    # Setup the default ahf_reader
    if ahf_reader is not None:
      self.ahf_reader = ahf_reader
    else:
      self.ahf_reader = galaxy_diver.read_ahf.AHFReader( self.kwargs['sdir'] )

    # Derived properties
    self.n_particles = self.particle_positions.shape[0]

  ########################################################################

  def find_ids( self, ids_to_return=[ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id', 'mt_halo_id', 'mt_gal_id'], galaxy_cut=0.1 ):
    '''Find relevant halo and galaxy IDs.

    Args:
      ids_to_return (list of strs): The types of id you want to get out.
      galaxy_cut (float): The fraction of the virial radius a particle must be inside to be counted as part of a galaxy.

    Returns:
      galaxy_and_halo_ids (dict): Keys are...
      Parameters:
        halo_id (np.array of ints): ID of the least-massive halo the particle is part of.
        host_halo_id (np.array of ints): ID of the host halo the particle is part of.
        gal_id (np.array of ints): ID of the smallest galaxy the particle is part of.
        host_gal_id (np.array of ints): ID of the host galaxy the particle is part of.
    '''

    # Load the ahf data
    self.ahf_reader.get_halos( self.kwargs['snum'] )

    # Dictionary to store the data in.
    galaxy_and_halo_ids = {}
    
    # Actually get the data
    for id_type in ids_to_return:
      if id_type == 'halo_id':
        galaxy_and_halo_ids['halo_id'] = self.find_halo_id()
      elif id_type == 'host_halo_id':
        galaxy_and_halo_ids['host_halo_id'] = self.find_host_id()
      elif id_type == 'gal_id':
        galaxy_and_halo_ids['gal_id'] = self.find_halo_id( galaxy_cut )
      elif id_type == 'host_gal_id':
        galaxy_and_halo_ids['host_gal_id'] = self.find_host_id( galaxy_cut )
      elif id_type == 'mt_halo_id':
        galaxy_and_halo_ids['mt_halo_id'] = self.find_halo_id( type_of_halo_id='mt_halo_id' )
      elif id_type == 'mt_gal_id':
        galaxy_and_halo_ids['mt_gal_id'] = self.find_halo_id( galaxy_cut, type_of_halo_id='mt_halo_id' )
      else:
        raise Exception( "Unrecognized id_type" )
    
    return galaxy_and_halo_ids

  ########################################################################

  def find_host_id( self, radial_cut_fraction=1. ):
    '''Find the host halos our particles are inside of some radial cut of.
    This is the host ID at a given snapshot, and is not the same as the merger tree halo ID.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*R_vir from the center.

    Returns:
      host_halo (np.array of ints): Shape ( n_particles, ). 
        The ID of the least massive substructure the particle's part of.
        If it's -1, then the halo ID is the host ID.
        If it's -2, then that particle is not part of any halo, within radial_cut_fraction*Rvir .
    '''

    # Get the halo ID
    halo_id = self.find_halo_id( radial_cut_fraction )

    # Get the host halo ID
    host_id = self.ahf_reader.ahf_halos['hostHalo'][ halo_id ]

    # Fix the invalid values (which come from not being associated with any halo)
    host_id_fixed = np.ma.fix_invalid( host_id, fill_value=-2 )

    return host_id_fixed.data.astype( int )

  ########################################################################

  def find_halo_id( self, radial_cut_fraction=1., type_of_halo_id='halo_id' ):
    '''Find the smallest halos our particles are inside of some radial cut of (we define this as the halo ID).
    In the case of using MT halo ID, we actually find the most massive our particles are inside some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*R_vir from the center.
      type_of_halo_id (str): If 'halo_id' then this is the halo_id at a given snapshot.
                             If 'mt_halo_id' then this is the halo_id according to the merger tree.

    Returns:
      halo_id (np.array of ints): Shape ( n_particles, ). 
        The ID of the least massive substructure the particle's part of.
        In the case of using the 'mt_halo_id', this is the ID of the most massive merger tree halo the particle's part of.
        If it's -2, then that particle is not part of any halo, within radial_cut_fraction*Rvir .
    '''

    # Choose parameters of the rest of the function based on what type of halo ID we're using
    if type_of_halo_id == 'halo_id':

      # Functions that change.
      find_containing_halos_fn = self.find_containing_halos
      arg_extremum_fn = np.argmin
      extremum_fn = np.min

      # Get the virial masses. It's okay to leave in comoving, since we're just finding the minimum
      m_vir = self.ahf_reader.ahf_halos['Mvir']

    elif type_of_halo_id == 'mt_halo_id':

      # Functions that change.
      find_containing_halos_fn = self.find_mt_containing_halos
      arg_extremum_fn = np.argmax
      extremum_fn = np.max

      # Get the virial masses. It's okay to leave in comoving, since we're just finding the maximum
      m_vir = self.ahf_reader.get_mtree_halo_quantity( quantity='Mvir', indice=self.kwargs['snum'], index=self.kwargs['mtree_halos_index'], tag='smooth' )

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
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*R_vir from the center.

    Returns:
      part_of_halo (np.array of bools): Shape (n_particles, n_halos). 
        If index [i, j] is True, then particle i is inside radial_cut_fraction*R_vir of the jth halo.
    '''

    # Get the halo positions
    halo_pos_comov = np.array([ self.ahf_reader.ahf_halos['Xc'], self.ahf_reader.ahf_halos['Yc'], self.ahf_reader.ahf_halos['Zc'] ]).transpose()
    halo_pos = halo_pos_comov/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']

    # Get the distances
    # Output is ordered such that dist[:,0] is the distance to the center of halo 0 for each particle
    dist = scipy.spatial.distance.cdist( self.particle_positions, halo_pos )

    # Get the radial distance
    r_vir_pkpc = self.ahf_reader.ahf_halos['Rvir']/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']
    radial_cut = radial_cut_fraction*r_vir_pkpc

    # Tile the radial cut to allow comparison with dist
    tiled_cut = np.tile( radial_cut, ( self.n_particles, 1 ) )

    # Find the halos that our particles are part of 
    part_of_halo = dist < tiled_cut

    return part_of_halo

  ########################################################################

  def find_mt_containing_halos( self, radial_cut_fraction=1. ):
    '''Find which MergerTrace halos our particles are inside of some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*R_vir from the center.

    Returns:
      part_of_halo (np.array of bools): Shape (n_particles, n_halos). 
        If index [i, j] is True, then particle i is inside radial_cut_fraction*R_vir of the jth halo, defined via the MergerTrace ID.
    '''

    # Load up the merger tree data
    self.ahf_reader.get_mtree_halos( self.kwargs['mtree_halos_index'], 'smooth' )

    part_of_halo = []
    for halo_id in self.ahf_reader.mtree_halos.keys():
      mtree_halo = self.ahf_reader.mtree_halos[ halo_id ]

      # Get the halo position
      halo_pos_comov = np.array([ mtree_halo['Xc'][ self.kwargs['snum'] ], mtree_halo['Yc'][ self.kwargs['snum'] ], mtree_halo['Zc'][ self.kwargs['snum'] ] ])
      halo_pos = halo_pos_comov/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']

      # Make halo_pos 2D for compatibility with cdist
      halo_pos = halo_pos[np.newaxis]

      # Get the distances
      dist = scipy.spatial.distance.cdist( self.particle_positions, halo_pos )

      # Get the radial distance
      r_vir_pkpc = mtree_halo['Rvir'][ self.kwargs['snum'] ]/( 1. + self.kwargs['redshift'] )/self.kwargs['hubble']
      radial_cut = radial_cut_fraction*r_vir_pkpc
      
      # Find if our particles are part of this halo
      part_of_this_halo = dist < radial_cut

      part_of_halo.append( part_of_this_halo )

    # Format part_of_halo correctly
    part_of_halo = np.array( part_of_halo ).transpose()[0]

    return part_of_halo
