#!/usr/bin/env python
'''Means to associate particles with galaxies at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

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

  def __init__( self ):
    pass

  ########################################################################

  def find_galaxies( self ):

    # Load the ahf data
    self.get_ahf_data()
    
    # Find the host halo for each particle
    self.find_host_halos()

    # Find the host galaxy, under the smallest galaxy definition
    self.find_host_gal_small()

    # Find the host galaxy, under the largest galaxy definition
    self.find_host_gal_large()

    return galaxy_associations

  ########################################################################



