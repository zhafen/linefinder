#!/usr/bin/env python
'''Means to associate particles with galaxies and halos at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os
import scipy
import time

import ahf_reading
import code_tools

########################################################################
########################################################################

class ParticleTrackGalaxyFinder( object ):
  '''Find the association with galaxies for entire particle tracks.'''

  def __init__( self, data_p ):
    '''Initialize.

    Args:
      data_p (dict): Includes...
        Required:
          sdir (str): Directory the AHF data is in.
          tracking_dir (str): Directory the ptrack data is in.
          tag (str): Identifying tag for the ptrack data
        Optional:
          ids_to_return (list of strs): The types of id you want to get out.
          galaxy_cut (float): Anything within galaxy_cut*R_vir is counted as being inside the virial radius. Defaults to 0.1.
    '''

    self.data_p = data_p

    code_tools.set_default_attribute( self, 'ids_to_return', [ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id'] )
    code_tools.set_default_attribute( self, 'galaxy_cut', 0.1 )

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
      data_p = {
        'redshift' : self.ptrack['redshift'][...][ i ],
        'snum' : self.ptrack['snum'][...][ i ],
        'hubble' : self.ptrack.attrs['hubble'],
        'sdir' : self.data_p['sdir'],
      }

      print 'Snapshot {}, P[redshift] = {:>7.3g}'.format( data_p['snum'], data_p['redshift'], )

      # Find the galaxy for a given snapshot
      galaxy_finder = GalaxyFinder( particle_positions, data_p )
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
    print "Using AHF data from this directory:\n    {}".format( self.data_p['sdir'] )
    print "Data will be saved here:\n    {}".format( self.data_p['tracking_dir'] )

    # Load the particle track data
    ptrack_filename = 'ptrack_{}.hdf5'.format( self.data_p['tag'] )
    self.ptrack_filepath = os.path.join( self.data_p['tracking_dir'], ptrack_filename )
    self.ptrack = h5py.File( self.ptrack_filepath, 'a' )

  ########################################################################

  def finish_data_processing( self ):
    '''Write the data, close the file, and print out information.'''

    # Get the number of particles, for use in reporting the time
    n_particles = self.ptrack[ 'rho' ][...].shape[0]

    # Save the data.
    for key in self.ptrack_gal_ids.keys():
      self.ptrack.create_dataset( key, data=self.ptrack_gal_ids[key] )
    self.ptrack.close()

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

  def __init__( self, particle_positions, data_p ):
    '''Initialize.

    Args:
      particle_positions (np.array): Positions with dimensions (n_particles, 3).
      data_p (dict): Includes...
        redshift (float): Redshift the particles are at.
        snum (int): Snapshot the particles correspond to.
        hubble (float): Cosmological hubble parameter (little h)
        sdir (str): Directory the AHF data is in.
    '''

    self.particle_positions = particle_positions
    self.data_p = data_p

    # Derived properties
    self.n_particles = self.particle_positions.shape[0]

  ########################################################################

  def find_ids( self, ids_to_return=[ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id'], galaxy_cut=0.1 ):
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
    self.ahf_reader = ahf_reading.AHFReader( self.data_p['sdir'] )
    self.ahf_reader.get_halos( self.data_p['snum'] )

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

    return host_id_fixed.data

  ########################################################################

  def find_halo_id( self, radial_cut_fraction=1. ):
    '''Find the smallest halos our particles are inside of some radial cut of (we define this as the halo ID).

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*R_vir from the center.

    Returns:
      halo_id (np.array of ints): Shape ( n_particles, ). 
        The ID of the least massive substructure the particle's part of.
        If it's -2, then that particle is not part of any halo, within radial_cut_fraction*Rvir .
    '''

    # Get the cut
    part_of_halo = self.find_containing_halos( radial_cut_fraction=radial_cut_fraction )

    # Get the virial masses. It's okay to leave in comoving, since we're just finding the minimum
    m_vir = self.ahf_reader.ahf_halos['Mvir']

    # Mask the data
    tiled_m_vir = np.tile( m_vir, ( self.n_particles, 1 ) )
    tiled_m_vir_ma = np.ma.masked_array( tiled_m_vir, mask=np.invert( part_of_halo ), )

    # Take the argmin of the masked data
    halo_id = tiled_m_vir_ma.argmin( 1 )
    
    # Account for the fact that the argmin defaults to 0 when there's nothing there
    mask = tiled_m_vir_ma.min( 1 ).mask
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
    halo_pos = halo_pos_comov/( 1. + self.data_p['redshift'] )/self.data_p['hubble']

    # Get the distances
    # Output is ordered such that dist[:,0] is the distance to the center of halo 0 for each particle
    dist = scipy.spatial.distance.cdist( self.particle_positions, halo_pos )

    # Get the radial distance
    r_vir_pkpc = self.ahf_reader.ahf_halos['Rvir']/( 1. + self.data_p['redshift'] )/self.data_p['hubble']
    radial_cut = radial_cut_fraction*r_vir_pkpc

    # Tile the radial cut to allow comparison with dist
    tiled_cut = np.tile( radial_cut, ( self.n_particles, 1 ) )

    # Find the halos that our particles are part of 
    part_of_halo = dist < tiled_cut

    return part_of_halo

