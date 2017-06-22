#!/usr/bin/env python
'''Tools for categorizing particles into different accretion modes.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os
import scipy.spatial
import sys
import time

import galaxy_diver.read_ahf
import astro_tools
import constants

########################################################################
########################################################################

class Classifier( object ):
  '''Loads the tracked particle data, and uses that to classify the particles into different categories.
  '''

  def __init__( self, **kwargs ):
    '''Setup the ID Finder.

    Keyword Args:
      Data Parameters:
        sdir (str): Data directory for AHF data.
        tracking_dir (str): Data directory for tracked particle data.
        tag (str): Identifying tag. Currently must be put in manually.

      Analysis Parameters:
        neg (int): Number of earliest indices for which we neglect accretion/ejection events.
                   If each indice corresponds to a snapshot, then it's the number of snapshots
        wind_vel_min (float): The minimum radial velocity (in km/s ) a particle must have to be considered ejection.
        wind_vel_min_vc (float): The minimum radial velocity (in units of the main galaxy circular velocity) a particle must have to be considered ejection.
        time_min (float): Minimum time (in Myr) a particle must reside in a galaxy to not count as pristine gas.
        time_interval_fac (float): Externally-processed mass is required to spend at least time_min during the
                                   interval time_interval_fac x time_min prior to accretion to qualify as a *merger*.
    '''

    self.kwargs = kwargs

  ########################################################################

  def classify_particles( self ):
    '''Run the full classification suite.'''

    # Print out starting information
    time_start = time.time()
    print "########################################################################"
    print "Starting Classifying!"
    print "########################################################################"
    print "Using tracked particle data from this directory:\n    {}".format( self.kwargs['tracking_dir'] )
    print "Using AHF data from this directory:\n    {}".format( self.kwargs['sdir'] )
    print "Data will be saved here:\n    {}".format( self.kwargs['tracking_dir'] )

    # Get the data files out
    self.read_data_files()

    # Do the auxiliary calculations
    print "Calculating radial velocity, circular velocity, and dt..."
    self.v_r = self.get_radial_velocity()
    self.v_c = self.get_circular_velocity()
    self.dt = self.get_time_difference()

    # Do the first wave of classifications
    print "Identifying accretion, ejection, etc..."
    self.is_in_other_gal = self.identify_is_in_other_gal()
    self.is_in_main_gal = self.identify_is_in_main_gal()
    self.gal_event_id = self.calc_gal_event_id()
    self.is_accreted = self.identify_accretion()
    self.is_ejected = self.identify_ejection()

    # Information on what happens before accretion.
    print "Figuring out what happens before first accretion..."
    self.is_before_first_acc = self.identify_is_before_first_acc()
    self.time_in_other_gal_before_acc = self.get_time_in_other_gal_before_acc()
    self.time_in_other_gal_before_acc_during_interval = self.get_time_in_other_gal_before_acc_during_interval()

    # Get the primary classifications
    print "Performing the main classifications..."
    self.is_pristine = self.identify_pristine()
    self.is_preprocessed = self.identify_preprocessed()
    self.is_mass_transfer = self.identify_mass_transfer()
    self.is_merger = self.identify_merger()
    self.is_wind = self.identify_wind()

    # Save the results
    self.save_classifications()

    # Print out end information
    time_end = time.time()
    print "########################################################################"
    print "Done Classifying!"
    print "########################################################################"
    print "Output file saved as:\n    {}".format( self.classification_filepath )
    print "Took {:.3g} seconds, or {:.3g} seconds per particle!".format( time_end - time_start, (time_end - time_start) / self.n_particle )

  ########################################################################

  def read_data_files( self ):
    '''Read the relevant data files, and store the data in a dictionary for easy access later on.'''

    print "Reading data..."

    self.ptrack = {}
    self.ptrack_attrs = {}
    def load_data_into_ptrack( filename ):

      filepath = os.path.join( self.kwargs['tracking_dir'], filename )
      f = h5py.File(filepath, 'r')

      # Store the particle track data in a dictionary
      for key in f.keys():
        self.ptrack[ key ] = f[ key ][...]

      # Store the ptrack attributes
      for key in f.attrs.keys():
        self.ptrack_attrs[ key ] = f.attrs[ key ]

      f.close()

    # Load Particle Tracking and Galaxy Finding Data
    ptrack_filename =  'ptrack_{}.hdf5'.format( self.kwargs['tag'] )
    galfind_filename =  'galfind_{}.hdf5'.format( self.kwargs['tag'] )
    load_data_into_ptrack( ptrack_filename )
    load_data_into_ptrack( galfind_filename )

    # Set useful state variables
    self.n_snap = self.ptrack['redshift'].size
    self.n_particle = self.ptrack['id'].size

    # Load the ahf data
    self.ahf_reader = galaxy_diver.read_ahf.AHFReader( self.kwargs['sdir'] )
    self.ahf_reader.get_mtree_halos( 'snum', 'smooth' )

  ########################################################################

  def save_classifications( self, classifications_to_save=[ 'is_pristine', 'is_preprocessed', 'is_merger', 'is_mass_transfer', 'is_wind' ] ):
    '''Save the results of running the classifier.

    Args:
      classifications_to_save (list of str): A list of the attributes you want to save.
    '''

    # Open up the file to save the data in.
    classification_filename =  'classified_' + self.kwargs['tag'] + '.hdf5'
    self.classification_filepath = os.path.join( self.kwargs['tracking_dir'], classification_filename )
    f = h5py.File( self.classification_filepath, 'a' )

    # Save the data
    for classification in classifications_to_save:

      data = getattr( self, classification )
      f.create_dataset( classification, data=data )

    f.close()

  ########################################################################
  # Auxilliary Calculations
  ########################################################################

  def get_radial_velocity( self ):
    '''Get the radial velocity of particles, relative to the main galaxy.

    Returns:
      v_r ( [n_particle, n_snap] np.array ) : The radial velocity of each particle at that redshift, relative to the main galaxy.
    '''

    # Get the position and velocity of the main galaxy
    main_mt_halo_p = self.ahf_reader.get_pos_or_vel( 'pos', self.ptrack_attrs[ 'main_mt_halo_id' ], self.ptrack[ 'snum' ] )
    main_mt_halo_v = self.ahf_reader.get_pos_or_vel( 'vel', self.ptrack_attrs[ 'main_mt_halo_id' ], self.ptrack[ 'snum' ] )

    # Apply cosmological corrections to the position of the main galaxy
    main_mt_halo_p *= 1./( 1. + self.ptrack['redshift'][:,np.newaxis] )/self.ptrack_attrs['hubble']
    # TODO: Check if the velocity also needs to be corrected (probably easiest is to compare to the velocity of the CoM, calculated by me)

    # Loop over each redshift
    v_r = []
    for i in range(self.n_snap):

      # Get the radial velocity of the particles
      v_r_i = scipy.spatial.distance.cdist( self.ptrack[ 'v' ][:,i], main_mt_halo_v[i][np.newaxis] ).flatten()

      # Get the radial distance of the particles for the hubble flow.
      r_i = scipy.spatial.distance.cdist( self.ptrack[ 'p' ][:,i], main_mt_halo_p[i][np.newaxis] ).flatten()

      # Add the hubble flow.
      hubble_factor = astro_tools.hubble_z( self.ptrack['redshift'][i], h=self.ptrack_attrs['hubble'], \
                                            omega_matter=self.ptrack_attrs['omega_matter'], omega_lambda=self.ptrack_attrs['omega_lambda'] )
      v_r_i += hubble_factor * r_i * constants.UNITLENGTH_IN_CM  / constants.UNITVELOCITY_IN_CM_PER_S  

      v_r.append( v_r_i )

    # Format the output
    v_r = np.array( v_r ).transpose()

    return v_r

  ########################################################################

  def get_circular_velocity( self ):
    '''Get the circular velocity of the halo.

    Returns:
      v_c : Circular velocity of the halo in km/s, indexed the same way that ahf_reader.mtree_halos[i] is.
    '''

    # Get the virial radius and mass of the main galaxy
    r_vir_kpc = self.ahf_reader.mtree_halos[0]['Rvir'][ self.ptrack[ 'snum' ] ]
    m_vir_msun = self.ahf_reader.mtree_halos[0]['Mvir'][ self.ptrack[ 'snum' ] ]
    
    # Convert r_vir and m_vir to physical units
    r_vir_kpc *= 1./( 1. + self.ptrack['redshift'] )/self.ptrack_attrs['hubble']
    m_vir_msun /= self.ptrack_attrs['hubble']

    # Convert r_vir and m_vir to cgs
    r_vir_cgs = r_vir_kpc*constants.CM_PER_KPC
    m_vir_cgs = m_vir_msun*constants.MSUN

    # Get v_c
    v_c_cgs = np.sqrt( constants.G_UNIV * m_vir_cgs / r_vir_cgs )

    # Convert to km/s
    v_c = v_c_cgs / constants.CM_PER_KM

    return v_c

  ########################################################################

  def get_time_difference( self ):
    '''Get the time between snapshots.

    Returns:
      dt ([n_particle, n_snap-1] np.array): Time between snapshots in Myr.
    '''

    # Replicate redshifts self.ptrack indexing (last one removed)
    redshift = np.tile( self.ptrack['redshift'][0:self.n_snap], (self.n_particle,1) )   

    # Age of the universe in Myr
    time = 1e3 * astro_tools.age_of_universe( redshift, h=self.ptrack_attrs['hubble'], omega_matter=self.ptrack_attrs['omega_matter'] )
    dt = time[:,:-1] - time[:,1:] 

    return dt

  ########################################################################
  # Auxilliary Classification Methods
  ########################################################################

  def identify_is_in_other_gal( self ):
    '''Identify what particles are in a galaxy besides the main galaxy.

    Returns:
      is_in_main_gal ( [n_particle, n_snap-1] np.array of bools) : True if in a galaxy other than the main galaxy at that redshift.
    '''

    # Get the ID of the main halo for a given snapshot (remember that the mtree halo ID isn't the same as the ID at a given snapshot).
    main_mtree_halo = self.ahf_reader.mtree_halos[ self.ptrack_attrs['main_mt_halo_id'] ]
    main_halo_id = main_mtree_halo[ 'ID' ][ self.ptrack[ 'snum' ] ]
    main_halo_id_tiled = np.tile( main_halo_id, ( self.n_particle, 1 ) )

    # Check if we're inside a galaxy other than the main galaxy
    # This step is necessary, and the inverse of it is not redundant, because it removes anything that's in the main halo *and* that's the least massive galaxy it's in.
    is_not_in_main_gal = ( self.ptrack['gal_id'] != main_halo_id_tiled )
    is_in_gal = ( self.ptrack['gal_id'] >= 0 )

    is_in_other_gal =  ( is_in_gal & is_not_in_main_gal )

    return is_in_other_gal

  ########################################################################

  def identify_is_in_main_gal( self ):
    '''Identify what particles are in a main galaxy. They must be in the main galaxy *and* not inside any other galaxy at that redshift, even a subhalo galaxy.

    Returns:
      is_in_main_gal ( [n_particle, n_snap-1] np.array of bools) : True if in the main galaxy (and not any other galaxy) at that redshift.
    '''

    is_not_in_other_gal = np.invert( self.is_in_other_gal )

    # If we're literally inside the main galaxy
    is_in_main_gal_literal = ( self.ptrack['mt_gal_id'][:,0:self.n_snap] == self.ptrack_attrs['main_mt_halo_id'] )

    # Find if particles are inside/outside of main galaxy at each redshift
    is_in_main_gal = ( is_in_main_gal_literal & is_not_in_other_gal )

    return is_in_main_gal

  ########################################################################

  def calc_gal_event_id( self ):
    '''Identify potential accretion/ejection events relative to main galaxy at each redshift

    Returns:
      gal_event_id ( [n_particle, n_snap-1] np.array of ints) : GalEvent = 0 (no change), 1 (entering galaxy), -1 (leaving galaxy) at that redshift
    '''

    # Find when the particles enter and exit the galaxy
    gal_event_id = self.is_in_main_gal[:,0:self.n_snap-1].astype( int ) - self.is_in_main_gal[:,1:self.n_snap].astype( int )

    return gal_event_id

  ########################################################################

  def identify_accretion( self ):
    '''Identify ALL gas/star accretion events, i.e. whether or not a particle was outside the galaxy at one redshift, and inside at the next'''

    is_accreted = ( self.gal_event_id == 1 )

    # Correct for "boundary conditions": neglect events at earliest snapshots
    is_accreted[:,-self.kwargs['neg']: ] = False

    return is_accreted

  ########################################################################

  def identify_ejection( self ):
    '''Identify ALL gas wind ejection events.
      These conditions must be met to identify as ejection:
        1. Inside the main galaxy at one snapshot, and not at the previous snapshot.
        2. Radial velocity of the particle relative to the main galaxy must be greater than some fraction of the circular velocity of the main galaxy.
        3. Radial velocity of the particle relative to the main galaxy must be greater than some base speed.
        4. The particle must be a gas particle.
        5. The particle must be outside any other galaxy.
    '''

    # Get the radial velocity out.
    v_r = self.get_radial_velocity()

    # Get the circular velocity out and tile it for comparison
    v_c = self.get_circular_velocity()
    v_c_tiled = np.tile( v_c, ( self.n_particle, 1 ) )

    # The conditions for being outside any galaxy
    is_outside_before_inside_after = ( self.gal_event_id == -1 ) # Condition 1
    has_minimum_vr_in_vc = ( v_r[:,0:self.n_snap-1] > self.kwargs['wind_vel_min_vc']*v_c_tiled[:,0:self.n_snap-1] ) # Condition 2
    has_minimum_vr = ( v_r[:,0:self.n_snap-1] > self.kwargs['wind_vel_min'] ) # Condition 3
    is_gas = ( self.ptrack['Ptype'][:,0:self.n_snap-1] == 0 ) # Condition 4
    is_outside_any_gal = ( self.ptrack['gal_id'][:,0:self.n_snap-1] < 0 ) # Condition 5

    is_ejected = ( 
      is_outside_before_inside_after & 
      has_minimum_vr_in_vc & 
      has_minimum_vr &
      is_gas & 
      is_outside_any_gal  
      )

    # Correct for "boundary conditions": neglect events at earliest snapshots
    is_ejected[:,-self.kwargs['neg']:] = False

    return is_ejected

  ########################################################################
  # What happens before accretion?
  ########################################################################

  def identify_is_before_first_acc( self ):
    '''Identify when before a particle's first accretion event.

    Returns:
      is_before_first_acc ([n_particle, n_snap-1] np.array of bools): If True, then the first accretion event for that particle hasn't happened yet.
    '''

    # Index to revert order of redshift snapshots
    ind_rev = np.arange( self.n_snap-2, -1, -1 )  

    # cumulative number of accretion events
    reverse_cum_num_acc =  self.is_accreted[:,ind_rev].cumsum(axis=1)
    cum_num_acc = reverse_cum_num_acc[:,ind_rev]      

    is_before_first_acc = ( cum_num_acc == 0 )  &  ( self.is_in_main_gal[:,0:self.n_snap-1] == 0 )

    return is_before_first_acc

  ########################################################################

  def get_time_in_other_gal_before_acc( self ):
    '''Get the amount of time in galaxies besides the main galaxy before being accreted.
    For a single time in another galaxy, this is the ( age of universe at the last snapshot before the conditions are true ) -
    ( age of the universe at the last snapshot where the conditions are true ), and generalizes to multiple events in other galaxies.

    Returns:
      time_in_other_gal_before_acc ([ n_particle, ] np.array of floats): Time in another galaxy before being first accreted onto the main galaxy.
    '''

    is_in_other_gal_before_first_acc = ( self.is_before_first_acc & self.is_in_other_gal[:,0:self.n_snap-1] )
    time_in_other_gal_before_acc = ( self.dt * is_in_other_gal_before_first_acc.astype( float )  ).sum(axis=1)

    return time_in_other_gal_before_acc

  ########################################################################

  def get_time_in_other_gal_before_acc_during_interval( self ):
    '''Get the amount of time in galaxies besides the main galaxy before being accreted, during an interval before being accreted.

    Returns:
      time_in_other_gal_before_acc_during_interval ([ n_particle, ] np.array of floats): Time in another galaxy before being first accreted onto the main galaxy,
        within some recent time interval
    '''

    # Get the total amount of time before being accreted
    cum_time_before_acc = ( self.dt * self.is_before_first_acc.astype( float ) ).cumsum(axis=1)

    # Conditions for counting up time
    time_interval = self.kwargs['time_interval_fac'] * self.kwargs['time_min']
    is_in_other_gal_in_time_interval_before_acc = (
      ( cum_time_before_acc <= time_interval ) & # Count up only the time before first accretion.
      self.is_before_first_acc & # Make sure we haven't accreted yet
      self.is_in_other_gal[:,0:self.n_snap-1] # Make sure we're in another galaxy at that time
      )

    time_in_other_gal_before_acc_during_interval = ( self.dt * is_in_other_gal_in_time_interval_before_acc.astype( float ) ).sum(axis=1)

    return time_in_other_gal_before_acc_during_interval

  ########################################################################
  # Main Classification Methods
  ########################################################################

  def identify_pristine( self ):
    '''Identify pristine gas, or "non-externally processed" gas.

    Returns:
      is_pristine ( [n_particle] np.array of bools ) : True for particle i if it has never spent some minimum amount of time in another galaxy before being accreted.
    '''

    is_pristine = ( self.time_in_other_gal_before_acc < self.kwargs['time_min'] )

    # Correct "boundary conditions": particles inside galaxy at earliest snapshot count as pristine
    for k in range( self.kwargs['neg'] ):
      is_pristine[ self.is_in_main_gal[:,self.n_snap-1-k] ] = True

    return is_pristine

  ########################################################################

  def identify_preprocessed( self ):
    '''Identify pre-proceesed gas, or "externally processed" gas.

    Returns:
      is_preprocessed ( [n_particle] np.array of bools ) : True for particle i if it has spent at least some minimum amount of time in another galaxy before being accreted.
    '''

    is_preprocessed = ( self.time_in_other_gal_before_acc >= self.kwargs['time_min'] )

    # Correct "boundary conditions": particles inside galaxy at earliest snapshot count as pristine
    for k in range( self.kwargs['neg'] ):
      is_preprocessed[ self.is_in_main_gal[:, self.n_snap-1-k] ] = False

    return is_preprocessed

  ########################################################################

  def identify_mass_transfer( self ):
    '''Boolean for whether or no particles are from mass transfer

    Returns:
      is_mass_transfer (np.array of bools) : True for particle i if it has been preprocessed but has *not*
        spent at least some minimum amount of time in another galaxy in a recent interval.
    '''
    has_not_spent_minimum_time = ( self.time_in_other_gal_before_acc_during_interval < self.kwargs['time_min'] )
    is_mass_transfer = (  self.is_preprocessed & has_not_spent_minimum_time )

    return is_mass_transfer

  ########################################################################


  def identify_merger( self ):
    '''Boolean for whether or no particles are from galaxies merging.

    Returns:
      is_merger ( [n_particle] np.array of bools ) : True for particle i if it has been preprocessed and has
        spent at least some minimum amount of time in another galaxy in a recent interval.
    '''
    has_spent_minimum_time = ( self.time_in_other_gal_before_acc_during_interval >= self.kwargs['time_min'] )
    is_merger = (  self.is_preprocessed & has_spent_minimum_time  )

    return is_merger

  ########################################################################

  def identify_wind( self ):
    '''Boolean for whether or not particles are from wind.

    Returns:
      is_wind ( [n_particle] np.array of bools ) : True for particle i if it has been ejected at least once before snapshot n 
    '''

    # Index to revert order of redshift snapshots
    ind_rev = np.arange( self.n_snap-2, -1, -1 )  

    # Cumulative number of ejection events
    cum_num_eject = self.is_ejected[:,ind_rev].cumsum( axis=1 )[:,ind_rev]      

    # Set up and build is_wind
    is_wind = np.zeros( ( self.n_particle, self.n_snap ), dtype=np.int32 )
    is_wind[:,0:self.n_snap-1] = ( cum_num_eject >= 1 )

    return is_wind.astype( bool )

