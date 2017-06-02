#!/usr/bin/env python
'''Means to associate particles with galaxies at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import scipy

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

  def __init__( self, particle_positions, redshift, hubble_param ):
    '''Initialize.

    Args:
      particle_positions (np.array): Positions with dimensions (n_particles, 3).
      redshift (float): Redshift the particles are at.
      hubble_param (float): Cosmological hubble parameter (little h)
    '''

    self.particle_positions = particle_positions
    self.redshift = redshift
    self.hubble_param = hubble_param

    # Derived properties
    self.n_particles = self.particle_positions.shape[0]

  ########################################################################

  def find_galaxies( self ):

    # Load the ahf data
    self.ahf_reader = ahf_reading.AHFReader( sdir )
    
    # Find the host halo for each particle
    self.find_host_halos()

    # Find the host galaxy, under the smallest galaxy definition
    self.find_host_gal_small()

    # Find the host galaxy, under the largest galaxy definition
    self.find_host_gal_large()

    return galaxy_associations

  ########################################################################

  def find_host_halos( self, ):

    pass

  ########################################################################

  def find_smallest_containing_halo( self, radial_cut_fraction=1. ):
    '''Find which halos our particles are inside of some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*R_vir from the center.

    Returns:
      smallest_containing_halo (np.array of ints): Shape ( n_particles, ). 
        If an int, it's the ID of the least massive substructure the particle's part of.
        If it's None, then that particle is not part of any halo, within radial_cut_fraction*Rvir .
    '''

    # Get the cut
    part_of_halo = self.find_containing_halos( radial_cut_fraction=radial_cut_fraction )

    # Get the virial masses. It's okay to leave in comoving, since we're just finding the minimum
    m_vir = self.ahf_reader.ahf_halos['Mvir']

    # Mask the data
    tiled_m_vir = np.tile( m_vir, ( self.n_particles, 1 ) )
    tiled_m_vir_ma = np.ma.masked_array( tiled_m_vir, mask=np.invert( part_of_halo ), )

    # Take the argmin of the masked data
    smallest_containing_halo = tiled_m_vir_ma.argmin( 1 )
    
    # Account for the fact that the argmin defaults to 0 when there's nothing there
    mask = tiled_m_vir_ma.min( 1 ).mask
    smallest_containing_halo = np.ma.filled( np.ma.masked_array(smallest_containing_halo, mask=mask) )

    return smallest_containing_halo

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
    halo_pos = halo_pos_comov/( 1. + self.redshift )/self.hubble_param

    # Get the distances
    # Output is ordered such that dist[:,0] is the distance to the center of halo 0 for each particle
    dist = scipy.spatial.distance.cdist( self.particle_positions, halo_pos )

    # Get the radial distance
    r_vir_pkpc = self.ahf_reader.ahf_halos['Rvir']/( 1. + self.redshift )/self.hubble_param
    radial_cut = radial_cut_fraction*r_vir_pkpc

    # Tile the radial cut to allow comparison with dist
    tiled_cut = np.tile( radial_cut, ( self.n_particles, 1 ) )

    # Find the halos that our particles are part of 
    part_of_halo = dist < tiled_cut

    return part_of_halo

