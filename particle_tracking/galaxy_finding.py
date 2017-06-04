#!/usr/bin/env python
'''Means to associate particles with galaxies at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import scipy

import ahf_reading

########################################################################
########################################################################

class ParticleTrackGalaxyFinder( object ):
  '''Find the association with galaxies for entire particle tracks.'''

  def __init__( self ):
    pass

  ########################################################################

  def find_galaxies_for_particle_tracks( self ):
    '''Main function.'''

    # Loop over each included snapshot.
    # TODO: Change this loop to a more appropriate loop
    for snum in snums:

      # Find the galaxy for a given snapshot
      galaxy_finder = GalaxyFinder()
      galaxy_associations = galaxy_finder.find_galaxies()

    # Save the data.
    self.save_galaxy_associations()

########################################################################
########################################################################

class GalaxyFinder( object ):
  '''Find the association with galaxies for a given set of particles at a given redshift.'''

  def __init__( self, particle_positions, data_p ):
    '''Initialize.

    Args:
      particle_positions (np.array): Positions with dimensions (n_particles, 3).
      data_p (dict): Includes...
        redshift (float): Redshift the particles are at.
        snum (int): Snapshot the particles correspond to.
        hubble_param (float): Cosmological hubble parameter (little h)
        sdir (str): Directory the AHF data is in.
    '''

    self.particle_positions = particle_positions
    self.data_p = data_p

    # Derived properties
    self.n_particles = self.particle_positions.shape[0]

  ########################################################################

  def find_ids( self, galaxy_cut=0.1 ):
    '''Find relevant halo and galaxy IDs.

    Args:
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
    self.ahf_reader.get_ahf_halos( self.data_p['snum'] )

    # Dictionary to store the data in.
    galaxy_and_halo_ids = {}
    
    # Actually get the data
    galaxy_and_halo_ids['halo_id'] = self.find_halo_id()
    galaxy_and_halo_ids['host_halo_id'] = self.find_host_id()
    galaxy_and_halo_ids['gal_id'] = self.find_halo_id( galaxy_cut )
    galaxy_and_halo_ids['host_gal_id'] = self.find_host_id( galaxy_cut )
    
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
    halo_pos = halo_pos_comov/( 1. + self.data_p['redshift'] )/self.data_p['hubble_param']

    # Get the distances
    # Output is ordered such that dist[:,0] is the distance to the center of halo 0 for each particle
    dist = scipy.spatial.distance.cdist( self.particle_positions, halo_pos )

    # Get the radial distance
    r_vir_pkpc = self.ahf_reader.ahf_halos['Rvir']/( 1. + self.data_p['redshift'] )/self.data_p['hubble_param']
    radial_cut = radial_cut_fraction*r_vir_pkpc

    # Tile the radial cut to allow comparison with dist
    tiled_cut = np.tile( radial_cut, ( self.n_particles, 1 ) )

    # Find the halos that our particles are part of 
    part_of_halo = dist < tiled_cut

    return part_of_halo

