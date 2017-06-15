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

import ahf_reading
import astro_tools
import code_tools
import constants

########################################################################
########################################################################

class Classifier( object ):
  '''Loads the tracked particle data, and uses that to classify the particles into different categories.
  '''

  def __init__( self, data_p ):
    '''Setup the ID Finder.

    Args:
      data_p (dict): Dictionary containing parameters for the analysis. Includes..
        Required:
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
            
          #types (list of ints): The particle data types to include.
          #snap_ini (int): Starting snapshot
          #snap_end (int): End snapshot
          #snap_step (int): How many snapshots to jump over?

      Analysis Parameters:
        Required:
          #target_ids (np.array): Array of IDs to target.
        Optional:
          #use_skid (bool): Whether or not to use skid, which is how things were originally set up.
          #target_child_ids (np.array): Array of child ids to target.
          #host_halo (bool): Whether or not to include halo data for tracked particles during the tracking process.
          #old_target_id_retrieval_method (bool): Whether or not to get the target ids in the way Daniel originally set up.
    '''

    self.data_p = data_p

  ########################################################################

  # TODO
  def classify_particles( self ):

    time_start = time.time()

    pass

  ########################################################################

  def read_data_files( self ):
    '''Read the relevant data files, and store the data in a dictionary for easy access later on.'''

    # Load Particle Tracking Data
    ptrack_filename =  'ptrack_' + self.data_p['tag'] + '.hdf5'
    ptrack_filepath = os.path.join( self.data_p['tracking_dir'], ptrack_filename )
    f = h5py.File(ptrack_filepath, 'r')

    # Store the particle track data in a dictionary
    self.ptrack = {}
    for key in f.keys():
      self.ptrack[ key ] = f[ key ][...]

    # Store the ptrack attributes
    self.ptrack_attrs = {}
    for key in f.attrs.keys():
      self.ptrack_attrs[ key ] = f.attrs[ key ]

    self.n_snap = self.ptrack['redshift'].size
    self.n_particle = self.ptrack['id'].size

    f.close()

    # Load the ahf data
    self.ahf_reader = ahf_reading.AHFReader( self.data_p['sdir'] )
    self.ahf_reader.get_mtree_halos( 'snum' )

    print 'Done with reading.'
    sys.stdout.flush()

  ########################################################################

  def simple_calculations( self ):
    '''Perform a number of simple calculations and store the results.
    '''

    ## --- coordinates wrt galaxy center (physical kpc)
    ## TODO AHF
    #r = self.ptrack['p'][:,0:n_snap,:] - skidgal['p_phi'][0:n_snap,:]

    ## --- radial distance to galaxy center (physical kpc)
    #R = np.sqrt((r*r).sum(axis=2))                   

    ## --- Hubble factor at all redshift
    #hubble_factor = astro_tools.hubble_z( self.ptrack['redshift'][0:n_snap], h=self.ptrack.attrs['hubble'], \
    #                                      omega_matter=self.ptrack.attrs['omega_matter'], omega_lambda=self.ptrack.attrs['omega_lambda'] )

    ## --- physical velocity wrt galaxy center (km/s)
    ## WARNING: need to update with v_phi??
    #v = f['v'][:,0:n_snap,:] - skidgal['v_CM'][0:n_snap,:]  +  hubble_factor[:,np.newaxis] * r * UnitLength_in_cm  / UnitVelocity_in_cm_per_s  

    ## --- radial velocity wrt galaxy center (km/s)
    #v_r = (v*r).sum(axis=2) / R 

    ## --- list of snapshots for indexing
    #snaplist = skidgal['snapnum'][0:n_snap] 

    ## --- index to revert order of redshift snapshots
    #ind_rev = np.arange(n_snap-2,-1,-1)  

    print '\nDone with initial calculations.'
    sys.stdout.flush()

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
    is_accreted[:,-self.data_p['neg']: ] = False

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
    has_minimum_vr_in_vc = ( v_r[:,0:self.n_snap-1] > self.data_p['wind_vel_min_vc']*v_c_tiled[:,0:self.n_snap-1] ) # Condition 2
    has_minimum_vr = ( v_r[:,0:self.n_snap-1] > self.data_p['wind_vel_min'] ) # Condition 3
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
    is_ejected[:,-self.data_p['neg']:] = False

    return is_ejected

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

    # TODO: delete?
    #IsFirstAcc = self.is_accreted  &  ( cum_num_acc == 1 )
    #IsJust_before_first_acc = np.roll( IsFirstAcc, 1, axis=1 );   IsJust_before_first_acc[:,0] = 0

  ########################################################################

  def get_time_in_other_gal_before_acc( self ):
    '''Get the amount of time in galaxies besides the main galaxy before being accreted.
    For a single time in another galaxy, this is the ( age of universe at the last snapshot before the conditions are true ) -
    ( age of the universe at the last snapshot where the conditions are true ), and generalizes to multiple events in other galaxies.

    Returns:
      time_in_other_gal_before_acc ([ n_particle, ] np.array of floats): Time in another galaxy before being first accreted onto the main galaxy.
    '''

    is_in_other_gal_before_first_acc = ( self.is_before_first_acc & self.is_in_other_gal[:,0:self.n_snap-1] )
    time_in_other_gal_before_acc = ( self.dt * is_in_other_gal_before_first_acc.astype( int )  ).sum(axis=1)

    return time_in_other_gal_before_acc

  ########################################################################

  def get_time_in_other_gal_before_acc_during_interval( self ):
    '''Get the amount of time in galaxies besides the main galaxy before being accreted, during an interval before being accreted.

    Returns:
      time_in_other_gal_before_acc_during_interval ([ n_particle, ] np.array of floats): Time in another galaxy before being first accreted onto the main galaxy,
        within some recent time interval
    '''

    # Get the total amount of time before being accreted
    cum_time_before_acc = ( self.dt * self.is_before_first_acc.astype(int) ).cumsum(axis=1)

    # Conditions for counting up time
    time_interval = self.data_p['time_interval_fac'] * self.data_p['time_min']
    is_in_other_gal_in_time_interval_before_acc = (
      ( cum_time_before_acc <= time_interval ) & # Count up only the time before first accretion.
      self.is_before_first_acc & # Make sure we haven't accreted yet
      self.is_in_other_gal[:,0:self.n_snap-1] # Make sure we're in another galaxy at that time
      )

    time_in_other_gal_before_acc_during_interval = ( self.dt * is_in_other_gal_in_time_interval_before_acc.astype(int) ).sum(axis=1)

    return time_in_other_gal_before_acc_during_interval

    # TODO: Remove?
    #TimeInOtherGal = ( dt * is_in_other_gal[:,0:n_snap-1].astype(int) ).sum(axis=1)

  ########################################################################
  # Main Classification Methods
  ########################################################################

  def identify_pristine( self ):
    '''Identify pristine gas, or "non-externally processed" gas.

    Returns:
      is_pristine ( [n_particle] np.array of bools ) : True for particle i if it has never spent some minimum amount of time in another galaxy before being accreted.
    '''

    is_pristine = ( self.time_in_other_gal_before_acc < self.data_p['time_min'] )

    # Correct "boundary conditions": particles inside galaxy at earliest snapshot count as pristine
    for k in range( self.data_p['neg'] ):
      is_pristine[ self.is_in_main_gal[:,self.n_snap-1-k] ] = True

    return is_pristine

  ########################################################################

  def identify_preprocessed( self ):
    '''Identify pre-proceesed gas, or "externally processed" gas.

    Returns:
      is_preprocessed ( [n_particle] np.array of bools ) : True for particle i if it has spent at least some minimum amount of time in another galaxy before being accreted.
    '''

    is_preprocessed = ( self.time_in_other_gal_before_acc >= self.data_p['time_min'] )

    # Correct "boundary conditions": particles inside galaxy at earliest snapshot count as pristine
    for k in range( self.data_p['neg'] ):
      is_preprocessed[ self.is_in_main_gal[:, self.n_snap-1-k] ] = False

    return is_preprocessed

  ########################################################################

  def identify_mass_transfer( self ):
    '''Boolean for whether or no particles are from mass transfer

    Returns:
      is_mass_transfer (np.array of bools) : True for particle i if it has been preprocessed but has *not*
        spent at least some minimum amount of time in another galaxy in a recent interval.
    '''
    has_not_spent_minimum_time = ( self.time_in_other_gal_before_acc_during_interval < self.data_p['time_min'] )
    is_mass_transfer = (  self.is_preprocessed & has_not_spent_minimum_time )

    return is_mass_transfer

  ########################################################################


  def identify_merger( self ):
    '''Boolean for whether or no particles are from galaxies merging.

    Returns:
      is_merger ( [n_particle] np.array of bools ) : True for particle i if it has been preprocessed and has
        spent at least some minimum amount of time in another galaxy in a recent interval.
    '''
    has_spent_minimum_time = ( self.time_in_other_gal_before_acc_during_interval >= self.data_p['time_min'] )
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
    is_wind[:,0:self.n_snap-1] = ( cum_num_eject >= 1 ).astype( bool )

    return is_wind

  ########################################################################
  # Old Stuff
  ########################################################################

  def old_acc_ej_mergers( self ):


    #IsInGalRe = ( R < GalDef*skidgal['ReStar'][0:n_snap] ).astype(int)

    GalEventRe = IsInGalRe[:,0:n_snap-1] - IsInGalRe[:,1:n_snap]


    n_eject = self.is_ejected.sum(axis=1)
    #Nacc = self.is_accreted.sum(axis=1)

    # --- identify FIRST accretion event
    cum_num_acc = self.is_accreted[:,ind_rev].cumsum(axis=1)[:,ind_rev]      # cumulative number of ACCRETION events

    IsFirstAcc = self.is_accreted  &  ( cum_num_acc == 1 )

    IsJustbefore_first_acc = np.roll( IsFirstAcc, 1, axis=1 );   IsJustbefore_first_acc[:,0] = 0

    is_before_first_acc = ( cum_num_acc == 0 )  &  ( is_in_main_gal[:,0:n_snap-1] == 0 )

    # --- identify LAST ejection event
    cum_num_eject = self.is_ejected[:,ind_rev].cumsum(axis=1)[:,ind_rev]      # cumulative number of EJECTION events

    cum_num_eject_rev = self.is_ejected.cumsum(axis=1)                           # cumulative number of EJECTION events (in reverse order!)
    IsLastEject = self.is_ejected  &  ( cum_num_eject_rev == 1 )

    # --- find star/gas particles inside the galaxy
    IsStarInside = is_in_main_gal  &  IsInGalRe  &  ( self.ptrack['Ptype'][:,0:n_snap]==4 )
    IsGasInside = is_in_main_gal  &  IsInGalRe  &  ( self.ptrack['Ptype'][:,0:n_snap]==0 )

    # --- identify STAR FORMATION event inside galaxy
    IsStarFormed = IsStarInside[:,0:n_snap-1]  &  ( self.ptrack['Ptype'][:,1:n_snap] == 0 )

    # --- identify GAS ACCRETION events
    IsGasAccreted = np.zeros( (n_particle,n_snap), dtype=np.int32 )
    IsGasFirstAcc = IsGasAccreted.copy()

    IsGasAccreted[:,0:n_snap-1] = self.is_accreted  &  ( self.ptrack['Ptype'][:,0:n_snap-1]==0 )   # initialize with all gas accretion events including "false" events 
    IsGasFirstAcc[:,0:n_snap-1] = IsFirstAcc  &  ( self.ptrack['Ptype'][:,0:n_snap-1]==0 )   # only the first accretion event

    Nacc = IsGasAccreted.sum(axis=1)

    ind = np.where( n_eject == 0 )[0]
    if ind.size > 0:
      IsGasAccreted[ind,:] = IsGasFirstAcc[ind,:]

    ind = np.where( (Nacc - n_eject > 1)  &  (n_eject > 0) )[0] 

    for i in ind:
       ind_eject = np.where( self.is_ejected[i,:] == 1 )[0]
       ind_acc = np.where( IsGasAccreted[i,:] == 1 )[0]
       IsGasAccreted[i,:] = IsGasFirstAcc[i,:]                             # initialize only to the first accretion event
       ind_this = np.searchsorted( ind_acc, ind_eject ) - 1
       ind_this = np.unique( ind_this[ ind_this>=0 ] )    
       IsGasAccreted[ i, ind_acc[ind_this] ] = 1
    Nacc = IsGasAccreted.sum(axis=1)


    print '\nDone with indexing.'
    sys.stdout.flush()

  ########################################################################

  def identify_host_dm_halo( self ):

    # --- identify the host dark matter halo

    HaloID = np.zeros(n_snap, dtype='int')
    Rvir = np.zeros(n_snap)
    Mvir = np.zeros(n_snap)
    nmaxh = 1000
    for ns in range(snaplist.size):
       halos = astro_tools.read_AHF_halos(simdir, snaplist[ns] )
       ind = np.where(IsStarInside[:,ns]==1)[0]
       if ind.size <= nmaxh:
         mode, count = stats.mode( self.ptrack['HaloID'][ ind, ns ] )
       else:
         np.random.seed(seed=1234)
         np.random.shuffle(ind)                                                       ###############################################################
         mode, count = stats.mode( self.ptrack['HaloID'][ np.sort(ind[0:nmaxh]), ns ] )           #######  WARNING CHECK THIS ###################################
       HaloID[ns] = mode[0]                                                           ##############################################################
       ind_halo = np.where( halos['id'] == HaloID[ns] )[0]
       if ind_halo.size != 1:
         print 'halo not found??  ', ns, snaplist[ns]
         continue
       Rvir[ns] = halos['Rvir'][ind_halo[0]]
       Mvir[ns] = halos['Mvir'][ind_halo[0]]
       

    print '\nDone with Dark matter halo.'
    sys.stdout.flush()

  ########################################################################

  def get_values_first_accretion( self ):
    '''Get values immediately before being accreted.
    '''

    mask = np.logical_not(IsJustbefore_first_acc)     # note that this may not be defined for all particles! 

    # --- redshift at fist accretion onto the galaxy
    redshift_FirstAcc = np.ma.masked_array( redshift[:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

    # --- temperature AT first accretion onto MAIN galaxy
    T_FirstAcc = np.ma.masked_array( self.ptrack['T'][:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

    # --- metallicity AT first accretion onto MAIN galaxy
    z_FirstAcc = np.ma.masked_array( self.ptrack['z'][:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

    # --- Ptype at first accretion onto the galaxy
    Ptype_FirstAcc = np.ma.masked_array( self.ptrack['Ptype'][:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)
    ind = np.where( Ptype_FirstAcc == -1 )[0]
    if ind.size > 0:
      Ptype_FirstAcc[ind] = self.ptrack['Ptype'][ind,n_snap]           # DAA: check n_snap here!!

    # --- Galaxy ID just prior to first accretion onto the main galaxy
    GalID_FirstAcc = np.ma.masked_array( self.ptrack['GalID'][:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

  ########################################################################

  def get_values_last_ejection( self ):
    '''Get values at last ejection.'''

    mask = np.logical_not(IsLastEject)

    # --- redshift at last ejection from the galaxy
    redshift_LastEject = np.ma.masked_array( redshift[:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

    # --- metallicity at last ejection
    z_LastEject = np.ma.masked_array( self.ptrack['z'][:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

    print '\nDone with first accretion.'
    sys.stdout.flush()

  ########################################################################

  def identify_wind_recycling( self ):

    all_DtEject = np.array([])
    all_RmaxEject = np.array([])
    all_RvirEject = np.array([])
    all_RvirEjectIni = np.array([])
    all_ReEject = np.array([])
    all_ReEjectIni = np.array([])
    all_TorbEjectIni = np.array([])
    all_MvirEject = np.array([])
    all_RedshiftEject = np.array([])
    all_RedshiftEjectIni = np.array([])
    all_SnapnumEject = np.array([])

    all_DtAcc = np.array([])
    all_RmaxAcc = np.array([])
    all_RvirAcc = np.array([])
    all_RvirAccIni = np.array([])
    all_ReAcc = np.array([])
    all_ReAccIni = np.array([])
    all_TorbAccIni = np.array([])
    all_MvirAcc = np.array([])
    all_RedshiftAcc = np.array([])
    all_RedshiftAccIni = np.array([])
    all_SnapnumAcc = np.array([])

    for i in range(n_particle):

       if (n_eject[i] < 1) or (Nacc[i] < 1):
         continue

       ind_eject = np.where( self.is_ejected[i,:] == 1 )[0]
       ind_acc = np.where( IsGasAccreted[i,:] == 1 )[0]
       if (ind_eject.size != n_eject[i]) or (ind_acc.size != Nacc[i]):
         print 'ueeeeehhhh!!!'
         continue

       EjectTimes = time[ i, ind_eject ]
       AccTimes = time[ i, ind_acc ]

       # --- from ejection to ejection ---
       if n_eject[i] >= 2:
         all_DtEject = np.append( all_DtEject, EjectTimes[:-1] - EjectTimes[1:] )
         all_RedshiftEject = np.append( all_RedshiftEject, redshift[ i, ind_eject[:-1] ] )              # redshift at re-ejection
         all_RedshiftEjectIni = np.append( all_RedshiftEjectIni, redshift[ i, ind_eject[1:] ] )         # redshift at ejection
         all_SnapnumEject = np.append( all_SnapnumEject, snaplist[ ind_eject[:-1] ] )
         for j in range(n_eject[i]-1):
            Rmax = np.max( R[ i, ind_eject[j]+1:ind_eject[j+1]+1 ] )
            all_RmaxEject = np.append( all_RmaxEject, Rmax )
            #all_RvirEject = np.append( all_RvirEject, np.mean( Rvir[ind_eject[j]+1:ind_eject[j+1]+1] ) )
            all_RvirEject = np.append( all_RvirEject, Rvir[ind_eject[j]+1] )
            all_RvirEjectIni = np.append( all_RvirEjectIni, Rvir[ind_eject[j+1]] )
            all_ReEject = np.append( all_ReEject, skidgal['ReStar'][ind_eject[j]+1] )
            all_ReEjectIni = np.append( all_ReEjectIni, skidgal['ReStar'][ind_eject[j+1]] )
            all_TorbEjectIni = np.append( all_TorbEjectIni, 2.*np.pi*(skidgal['Rhalf'][ind_eject[j+1]] / skidgal['VcRhalf'][ind_eject[j+1]]) * CM_PER_KPC/CM_PER_KM/SEC_PER_YEAR/1e6 )
            all_MvirEject = np.append( all_MvirEject, np.mean( Mvir[ind_eject[j]+1:ind_eject[j+1]+1] ) )

       # --- from ejection to accretion ---
       if ind_acc[0] < ind_eject[0]:
         all_DtAcc = np.append( all_DtAcc, AccTimes[0:n_eject[i]] - EjectTimes[:] )
         all_RedshiftAcc = np.append( all_RedshiftAcc, redshift[ i, ind_acc[0:n_eject[i]] ] )    # redshift at re-accretion
         all_RedshiftAccIni = np.append( all_RedshiftAccIni, redshift[ i, ind_eject ] )         # redshift at ejection
         all_SnapnumAcc = np.append( all_SnapnumAcc, snaplist[ ind_acc[0:n_eject[i]] ] )
         for j in range(n_eject[i]):
            Rmax = np.max( R[ i, ind_acc[j]:ind_eject[j]+1 ] )
            all_RmaxAcc = np.append( all_RmaxAcc, Rmax )
            #all_RvirAcc = np.append( all_RvirAcc, np.mean( Rvir[ind_acc[j]:ind_eject[j]+1] ) )
            all_RvirAcc = np.append( all_RvirAcc, Rvir[ind_acc[j]] )
            all_RvirAccIni = np.append( all_RvirAccIni, Rvir[ind_eject[j]] )
            all_ReAcc = np.append( all_ReAcc, skidgal['ReStar'][ind_acc[j]] )
            all_ReAccIni = np.append( all_ReAccIni, skidgal['ReStar'][ind_eject[j]] )
            all_TorbAccIni = np.append( all_TorbAccIni, 2.*np.pi*(skidgal['Rhalf'][ind_eject[j]] / skidgal['VcRhalf'][ind_eject[j]]) * CM_PER_KPC/CM_PER_KM/SEC_PER_YEAR/1e6 )
            all_MvirAcc = np.append( all_MvirAcc, np.mean( Mvir[ind_acc[j]:ind_eject[j]+1] ) )
       elif n_eject[i] >= 2:
         all_DtAcc = np.append( all_DtAcc, AccTimes[0:n_eject[i]-1] - EjectTimes[1:] )
         all_RedshiftAcc = np.append( all_RedshiftAcc, redshift[ i, ind_acc[0:n_eject[i]-1] ] )   # redshift at re-accretion
         all_RedshiftAccIni = np.append( all_RedshiftAccIni, redshift[ i, ind_eject[1:] ] )      # redshift at ejection
         all_SnapnumAcc = np.append( all_SnapnumAcc, snaplist[ ind_acc[0:n_eject[i]-1] ] )
         for j in range(n_eject[i]-1):
            Rmax = np.max( R[ i, ind_acc[j]:ind_eject[j+1]+1 ] )
            all_RmaxAcc = np.append( all_RmaxAcc, Rmax )
            #all_RvirAcc = np.append( all_RvirAcc, np.mean( Rvir[ind_acc[j]:ind_eject[j+1]+1] ) )
            all_RvirAcc = np.append( all_RvirAcc, Rvir[ind_acc[j]] )
            all_RvirAccIni = np.append( all_RvirAccIni, Rvir[ind_eject[j+1]] )
            all_ReAcc = np.append( all_ReAcc, skidgal['ReStar'][ind_acc[j]] )
            all_ReAccIni = np.append( all_ReAccIni, skidgal['ReStar'][ind_eject[j+1]] )
            all_TorbAccIni = np.append( all_TorbAccIni, 2.*np.pi*(skidgal['Rhalf'][ind_eject[j+1]] / skidgal['VcRhalf'][ind_eject[j+1]]) * CM_PER_KPC/CM_PER_KM/SEC_PER_YEAR/1e6 )
            all_MvirAcc = np.append( all_MvirAcc, np.mean( Mvir[ind_acc[j]:ind_eject[j+1]+1] ) )

    if all_DtEject.min() < 0:
      print '....... problems with all_DtEject .......'
    if all_DtAcc.min() < 0:
      print '....... problems with all_DtAcc .......'

    print '\nDone with wind recycling.'
    sys.stdout.flush()

  ########################################################################

  def other_accretion_modes( self ):

    #--- maximum temperature reached OUTSIDE of ANY galaxy  (T values are "invalid" when the particle is inside any galaxy)
    mask = np.logical_not(IsOutsideAnyGal)
    TmaxOutside = np.ma.masked_array( self.ptrack['T'][:,0:n_snap], mask=mask ).max(axis=1).filled(fill_value =-1)

    #--- maximum temperature reached BEFORE first accretion onto MAIN galaxy and OUTSIDE of any galaxy  (T values are "invalid" after first accretion)
    mask = np.logical_not( is_before_first_acc & IsOutsideAnyGal[:,0:n_snap-1] )
    TmaxBeforeAcc = np.ma.masked_array( self.ptrack['T'][:,0:n_snap-1], mask=mask ).max(axis=1).filled(fill_value =-1)
    
    #--- COLD vs HOT 

    Tmax = TmaxBeforeAcc

    IsHotMode = ( Tmax > ColdHotDef ).astype(int)
    IsColdMode = ( Tmax <= ColdHotDef ).astype(int)

    IsHotMode_mask = np.tile( IsHotMode[:,np.newaxis], n_snap )
    IsColdMode_mask = np.tile( IsColdMode[:,np.newaxis], n_snap )


    #correct for "boundary conditions": particles inside galaxy at earliest snapshots cannot count as winds
    #for k in range(4):
    #   is_wind[ is_in_main_gal[:,n_snap-1-k] == 1, n_snap-1-k ] = 0

  ########################################################################

  def stellar_mass_in_different_modes( self ):

    Nstar = IsStarInside.sum(axis=0)

    StarMass = ( self.ptrack['m'][:,0:n_snap] * IsStarInside ).sum(axis=0)
    StarMassHot = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * IsHotMode_mask ).sum(axis=0)
    StarMassCold = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * IsColdMode_mask ).sum(axis=0)
    StarMassWind = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_wind ).sum(axis=0)
    StarMassWindHot = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_wind * IsHotMode_mask ).sum(axis=0)
    StarMassWindCold = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_wind * IsColdMode_mask ).sum(axis=0)

    GasMass = ( self.ptrack['m'][:,0:n_snap] * IsGasInside ).sum(axis=0)
    GasMassWind = ( self.ptrack['m'][:,0:n_snap] * IsGasInside * is_wind ).sum(axis=0)

    Sfr = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside ).sum(axis=0)
    SfrHot = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * IsHotMode_mask ).sum(axis=0)
    SfrCold = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * IsColdMode_mask ).sum(axis=0)
    SfrWind = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_wind ).sum(axis=0)
    SfrWindHot = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_wind * IsHotMode_mask ).sum(axis=0)
    SfrWindCold = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_wind * IsColdMode_mask ).sum(axis=0)

    # --- "Msun per snapshot" from gas accretion events
    AccretedGasMass = ( self.ptrack['m'][:,0:n_snap] * IsGasAccreted ).sum(axis=0)
    AccretedGasMassWind = ( self.ptrack['m'][:,0:n_snap] * IsGasAccreted * is_wind ).sum(axis=0) 

  ########################################################################

  def other_growth_modes( self ):

    # Woot, moved everything I need out of here.

    pass




  ########################################################################

  def add_additional_info_to_preprocessed_mode( self ):

    is_preprocessed_mask = np.tile( is_preprocessed[:,np.newaxis], n_snap )
    is_mass_transfer_mask = np.tile( is_mass_transfer[:,np.newaxis], n_snap )
    is_merger_mask = np.tile( is_merger[:,np.newaxis], n_snap )
    IsStarAcc_mask = np.tile( (Ptype_FirstAcc == 4).astype(int)[:,np.newaxis], n_snap )
    IsGasAcc_mask = np.tile( (Ptype_FirstAcc == 0).astype(int)[:,np.newaxis], n_snap )

    StarMassFromPreProcessed = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_preprocessed_mask ).sum(axis=0)
    StarMassFromMassTransfer = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_mass_transfer_mask ).sum(axis=0)
    StarMassFromMassTransferGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_mass_transfer_mask * IsGasAcc_mask ).sum(axis=0)
    StarMassFromMassTransferStar = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_mass_transfer_mask * IsStarAcc_mask ).sum(axis=0)
    StarMassFromMerger = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_merger_mask ).sum(axis=0)
    StarMassFromMergerGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_merger_mask * IsGasAcc_mask ).sum(axis=0)
    StarMassFromMergerStar = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_merger_mask * IsStarAcc_mask ).sum(axis=0)

    GasMassFromPreProcessed = ( self.ptrack['m'][:,0:n_snap] * IsGasInside * is_preprocessed_mask ).sum(axis=0)
    GasMassFromMassTransfer = ( self.ptrack['m'][:,0:n_snap] * IsGasInside * is_mass_transfer_mask ).sum(axis=0)
    GasMassFromMerger = ( self.ptrack['m'][:,0:n_snap] * IsGasInside * is_merger_mask ).sum(axis=0)

    SfrFromPreProcessed = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_preprocessed_mask ).sum(axis=0) 
    SfrFromMassTransfer = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_mass_transfer_mask ).sum(axis=0)
    SfrFromMerger = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_merger_mask ).sum(axis=0)

    # --- "Msun per snapshot" from gas accretion events
    AccretedGasMassFromPreProcessed = ( self.ptrack['m'][:,0:n_snap] * IsGasAccreted * is_preprocessed_mask ).sum(axis=0)  
    AccretedGasMassFromTransfer = ( self.ptrack['m'][:,0:n_snap] * IsGasAccreted * is_mass_transfer_mask ).sum(axis=0)
    AccretedGasMassFromMerger = ( self.ptrack['m'][:,0:n_snap] * IsGasAccreted * is_merger_mask ).sum(axis=0)

  ########################################################################

  def add_additional_information_pristine_mode( self ):

    is_pristine_mask = np.tile( is_pristine[:,np.newaxis], n_snap )

    StarMassFromPristine = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask ).sum(axis=0)
    StarMassFromPristineHot = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * IsHotMode_mask ).sum(axis=0)
    StarMassFromPristineCold = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * IsColdMode_mask ).sum(axis=0)
    StarMassFromPristineWind = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * is_wind ).sum(axis=0)
    StarMassFromPristineWindHot = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * is_wind * IsHotMode_mask ).sum(axis=0)
    StarMassFromPristineWindCold = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * is_wind * IsColdMode_mask ).sum(axis=0)

    StarMassFromPristineGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * IsGasAcc_mask ).sum(axis=0)
    StarMassFromPristineHotGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * IsHotMode_mask * IsGasAcc_mask ).sum(axis=0)
    StarMassFromPristineColdGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * IsColdMode_mask * IsGasAcc_mask ).sum(axis=0)
    StarMassFromPristineWindGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * is_wind * IsGasAcc_mask ).sum(axis=0)
    StarMassFromPristineWindHotGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * is_wind * IsHotMode_mask * IsGasAcc_mask ).sum(axis=0)
    StarMassFromPristineWindColdGas = ( self.ptrack['m'][:,0:n_snap] * IsStarInside * is_pristine_mask * is_wind * IsColdMode_mask * IsGasAcc_mask ).sum(axis=0)

    GasMassFromPristine = ( self.ptrack['m'][:,0:n_snap] * IsGasInside * is_pristine_mask ).sum(axis=0)
    GasMassFromPristineWind = ( self.ptrack['m'][:,0:n_snap] * IsGasInside * is_pristine_mask * is_wind ).sum(axis=0)

    SfrFromPristine = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_pristine_mask ).sum(axis=0)
    SfrFromPristineHot = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_pristine_mask * IsHotMode_mask ).sum(axis=0)
    SfrFromPristineCold = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_pristine_mask * IsColdMode_mask ).sum(axis=0)
    SfrFromPristineWind = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_pristine_mask * is_wind ).sum(axis=0)
    SfrFromPristineWindHot = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_pristine_mask * is_wind * IsHotMode_mask ).sum(axis=0)
    SfrFromPristineWindCold = ( self.ptrack['sfr'][:,0:n_snap] * IsGasInside * is_pristine_mask * is_wind * IsColdMode_mask ).sum(axis=0)

    # --- "Msun per snapshot" from gas accretion events
    AccretedGasMassFromPristine = ( self.ptrack['m'][:,0:n_snap] * IsGasAccreted * is_pristine_mask ).sum(axis=0)
    AccretedGasMassFromPristineWind = ( self.ptrack['m'][:,0:n_snap] * IsGasAccreted * is_pristine_mask * is_wind ).sum(axis=0)

  ########################################################################

  def calculate_mass_loading( self ):

    IsLost = np.tile( (is_in_main_gal[:,0]==0).astype(int)[:,np.newaxis], n_snap-1 )

    GasMassLost = ( self.ptrack['m'][:,0:n_snap-1] * IsLastEject * IsLost ).sum(axis=0)

    MetalMassLost = ( self.ptrack['m'][:,0:n_snap-1] * self.ptrack['z'][:,0:n_snap-1] * IsLastEject * IsLost ).sum(axis=0) * SolarAbundance

    GasMassEjected = ( self.ptrack['m'][:,0:n_snap-1] * self.is_ejected  ).sum(axis=0)

    StarMassFormed = ( self.ptrack['m'][:,0:n_snap-1] * IsStarFormed ).sum(axis=0)


    print '\nDone with accretion modes.'
    sys.stdout.flush()

  ########################################################################

  # TODO
  def save_data( self ):

    pass

########################################################################
########################################################################

def delete_me_when_done():

  #######################################
   ### DEFINE SIMULATION DIRECTORIES ###
  #######################################

  sim_list = [ #'m13_mr_Dec16_2013'
               'm12v_mr_Dec5_2013_3'
               #'B1_hr_Dec5_2013_11'
               #'m12qq_hr_Dec16_2013'
               #'m11_hhr_Jan9_2013'
               #'m10_hr_Dec9_2013'
               #, 'm09_hr_Dec16_2013'
             ]

  """
  sim_list = [ 'm13_mr_Dec16_2013',
               'm12v_mr_Dec5_2013_3',
               'B1_hr_Dec5_2013_11',
               'm12qq_hr_Dec16_2013',
               'm11_hhr_Jan9_2013',
               'm10_hr_Dec9_2013'
               #, 'm09_hr_Dec16_2013'
             ]
  """

  tag = 'n5'
  #tag = 'all'



  GalDef = 2                  # imposed size of SKID galaxy in units of the stellar effective radius 
  wind_vel_min = 15             # minimum absolute radial velocity to be considered wind in km/s
  wind_vel_min_vc_frac = 2            # minimum radial velocity to be considered wind in units of the maximum circular velocity
  ColdHotDef = 2.5e5          # Temperature (k) threshold to separate COLD/HOT modes
  time_min = 100.              # Minimum time (Myr) spent in other galaxy prior to first accretion to qualify as EXTERNALLY-PROCESSED contribution 
  time_interval_fac = 5.        # Externally-processed mass is required to spend at least time_min during the interval time_interval_fac x time_min prior to accretion to qualify as MERGER
  neg = 5                     # Number of earliest snapshots for which we neglect accretion/ejection events


  outname = 'accmode_idlist_%s_g%dv%dvc%dt%dti%dneg%d.hdf5' % ( tag, GalDef, wind_vel_min, wind_vel_min_vc_frac, time_min, time_interval_fac, neg )


  skipreading = 0

  # TODO: Remove. This should be obsolete.
  #simdir = '/scratch/02089/anglesd/FIRE/' + sname + '/'
  #tracking_dir = '/work/02089/anglesd/FIRE/' + sname + '/'

  #########################
   ### READ DATA FILES ###
  #########################

  #########################
   ### CALCULATE STUFF ###
  #########################

  ###############################################
   ### IDENTIFY ACCRETION, EJECTION, MERGERS ###
  ###############################################

  #########################
  ### DARK MATTER HALO ###
  #########################

  ####################################
  ### JUST BEFORE FIRST ACCRETION ###
  ####################################

  #######################
   ### LAST EJECTION ###
  #######################

  ########################
   ### WIND RECYCLING ###
  ########################

  ##################################
   ### SEPARATE ACCRETION MODES ###
  ##################################

  ###################################################
   ### TOTAL STELLAR MASS IN EACH ACCRETION MODE ###
  ###################################################

  ###############################
   ### SEPARATE GROWTH MODES ###
  ###############################

  ############################################
   ### all specific to PRE-PROCESSED MODE ###
  ############################################

  #######################################
   ### all specific to PRISTINE MODE ###
  #######################################

  ####################################
   ### MASS LOADINGS -- MASS LOST ###
  ####################################

  ##############################
   ### SAVE RESULTS TO FILE ###
  ##############################

  #outname = 'accmode_' + ptrack_filepath + '-' + sname + '.hdf5'
  #outname = 'accmode_idlist_' + tag + '.hdf5'

  if os.path.isfile(tracking_dir + outname):
    os.remove(tracking_dir + outname)

  outf = h5py.File(tracking_dir + outname, 'w')

  #outf.create_dataset('redshift', data=f['redshift'][0:n_snap])
  outf.create_dataset('redshift', data=redshift[0,:])
  outf.create_dataset('dt', data=dt[0,:])

  outf.create_dataset('R', data=R)
  outf.create_dataset('v_r', data=v_r)

  outf.create_dataset('is_in_main_gal', data=is_in_main_gal)
  outf.create_dataset('IsInGalRe', data=IsInGalRe)
  outf.create_dataset('IsStarInside', data=IsStarInside)
  outf.create_dataset('IsGasInside', data=IsGasInside)
  outf.create_dataset('self.is_ejected', data=self.is_ejected)
  outf.create_dataset('self.is_accreted', data=self.is_accreted)
  outf.create_dataset('IsGasAccreted', data=IsGasAccreted)

  outf.create_dataset('is_pristine', data=is_pristine)
  outf.create_dataset('is_preprocessed', data=is_preprocessed)
  outf.create_dataset('is_mass_transfer', data=is_mass_transfer)
  outf.create_dataset('is_merger', data=is_merger)
  outf.create_dataset('IsHotMode', data=IsHotMode)
  outf.create_dataset('IsColdMode', data=IsColdMode)
  outf.create_dataset('is_wind', data=is_wind)


  outf.create_dataset('TmaxOutside', data=TmaxOutside)
  outf.create_dataset('TmaxBeforeAcc', data=TmaxBeforeAcc)
  outf.create_dataset('n_eject', data=n_eject)
  outf.create_dataset('Nacc', data=Nacc)
  outf.create_dataset('TimeInOtherGal', data=TimeInOtherGal)
  outf.create_dataset('time_in_other_gal_before_acc', data=time_in_other_gal_before_acc)
  outf.create_dataset('time_in_other_gal_before_acc_during_interval', data=time_in_other_gal_before_acc_during_interval)

  outf.create_dataset('T_FirstAcc', data=T_FirstAcc)
  outf.create_dataset('redshift_FirstAcc', data=redshift_FirstAcc)
  outf.create_dataset('z_FirstAcc', data=z_FirstAcc)
  outf.create_dataset('Ptype_FirstAcc', data=Ptype_FirstAcc)
  outf.create_dataset('GalID_FirstAcc', data=GalID_FirstAcc)

  outf.create_dataset('redshift_LastEject', data=redshift_LastEject)
  outf.create_dataset('z_LastEject', data=z_LastEject)

  outf.create_dataset('all_DtEject', data=all_DtEject)
  outf.create_dataset('all_RmaxEject', data=all_RmaxEject)
  outf.create_dataset('all_RvirEject', data=all_RvirEject)
  outf.create_dataset('all_RvirEjectIni', data=all_RvirEjectIni)
  outf.create_dataset('all_ReEject', data=all_ReEject)
  outf.create_dataset('all_ReEjectIni', data=all_ReEjectIni)
  outf.create_dataset('all_TorbEjectIni', data=all_TorbEjectIni)
  outf.create_dataset('all_MvirEject', data=all_MvirEject)
  outf.create_dataset('all_RedshiftEject', data=all_RedshiftEject)
  outf.create_dataset('all_RedshiftEjectIni', data=all_RedshiftEjectIni)
  outf.create_dataset('all_SnapnumEject', data=all_SnapnumEject)
  outf.create_dataset('all_DtAcc', data=all_DtAcc)
  outf.create_dataset('all_RmaxAcc', data=all_RmaxAcc)
  outf.create_dataset('all_RvirAcc', data=all_RvirAcc)
  outf.create_dataset('all_RvirAccIni', data=all_RvirAccIni)
  outf.create_dataset('all_ReAcc', data=all_ReAcc)
  outf.create_dataset('all_ReAccIni', data=all_ReAccIni)
  outf.create_dataset('all_TorbAccIni', data=all_TorbAccIni)
  outf.create_dataset('all_MvirAcc', data=all_MvirAcc)
  outf.create_dataset('all_RedshiftAcc', data=all_RedshiftAcc)
  outf.create_dataset('all_RedshiftAccIni', data=all_RedshiftAccIni)
  outf.create_dataset('all_SnapnumAcc', data=all_SnapnumAcc)

  outf.create_dataset('Nstar', data=Nstar)

  outf.create_dataset('StarMass', data=StarMass) 
  outf.create_dataset('StarMassHot', data=StarMassHot)
  outf.create_dataset('StarMassCold', data=StarMassCold)
  outf.create_dataset('StarMassWind', data=StarMassWind)
  outf.create_dataset('StarMassWindHot', data=StarMassWindHot)
  outf.create_dataset('StarMassWindCold', data=StarMassWindCold)

  outf.create_dataset('GasMass', data=GasMass)
  outf.create_dataset('GasMassWind', data=GasMassWind)

  outf.create_dataset('Sfr', data=Sfr)
  outf.create_dataset('SfrHot', data=SfrHot)
  outf.create_dataset('SfrCold', data=SfrCold)
  outf.create_dataset('SfrWind', data=SfrWind)
  outf.create_dataset('SfrWindHot', data=SfrWindHot)
  outf.create_dataset('SfrWindCold', data=SfrWindCold)

  outf.create_dataset('StarMassFromPreProcessed', data=StarMassFromPreProcessed)
  outf.create_dataset('StarMassFromMassTransfer', data=StarMassFromMassTransfer)
  outf.create_dataset('StarMassFromMassTransferGas', data=StarMassFromMassTransferGas)
  outf.create_dataset('StarMassFromMassTransferStar', data=StarMassFromMassTransferStar)
  outf.create_dataset('StarMassFromMerger', data=StarMassFromMerger)
  outf.create_dataset('StarMassFromMergerGas', data=StarMassFromMergerGas) 
  outf.create_dataset('StarMassFromMergerStar', data=StarMassFromMergerStar)

  outf.create_dataset('GasMassFromPreProcessed', data=GasMassFromPreProcessed)
  outf.create_dataset('GasMassFromMassTransfer', data=GasMassFromMassTransfer)
  outf.create_dataset('GasMassFromMerger', data=GasMassFromMerger)

  outf.create_dataset('SfrFromPreProcessed', data=SfrFromPreProcessed)
  outf.create_dataset('SfrFromMassTransfer', data=SfrFromMassTransfer)
  outf.create_dataset('SfrFromMerger', data=SfrFromMerger)

  outf.create_dataset('StarMassFromPristine', data=StarMassFromPristine)
  outf.create_dataset('StarMassFromPristineHot', data=StarMassFromPristineHot)
  outf.create_dataset('StarMassFromPristineCold', data=StarMassFromPristineCold)
  outf.create_dataset('StarMassFromPristineWind', data=StarMassFromPristineWind)
  outf.create_dataset('StarMassFromPristineWindHot', data=StarMassFromPristineWindHot)
  outf.create_dataset('StarMassFromPristineWindCold', data=StarMassFromPristineWindCold)

  outf.create_dataset('StarMassFromPristineGas', data=StarMassFromPristineGas)
  outf.create_dataset('StarMassFromPristineHotGas', data=StarMassFromPristineHotGas)
  outf.create_dataset('StarMassFromPristineColdGas', data=StarMassFromPristineColdGas)
  outf.create_dataset('StarMassFromPristineWindGas', data=StarMassFromPristineWindGas)
  outf.create_dataset('StarMassFromPristineWindHotGas', data=StarMassFromPristineWindHotGas)
  outf.create_dataset('StarMassFromPristineWindColdGas', data=StarMassFromPristineWindColdGas)

  outf.create_dataset('GasMassFromPristine', data=GasMassFromPristine)
  outf.create_dataset('GasMassFromPristineWind', data=GasMassFromPristineWind)

  outf.create_dataset('SfrFromPristine', data=SfrFromPristine)            
  outf.create_dataset('SfrFromPristineHot', data=SfrFromPristineHot)
  outf.create_dataset('SfrFromPristineCold', data=SfrFromPristineCold)
  outf.create_dataset('SfrFromPristineWind', data=SfrFromPristineWind)
  outf.create_dataset('SfrFromPristineWindHot', data=SfrFromPristineWindHot)
  outf.create_dataset('SfrFromPristineWindCold', data=SfrFromPristineWindCold)

  outf.create_dataset('AccretedGasMass', data=AccretedGasMass)
  outf.create_dataset('AccretedGasMassWind', data=AccretedGasMassWind)
  outf.create_dataset('AccretedGasMassFromPreProcessed', data=AccretedGasMassFromPreProcessed)
  outf.create_dataset('AccretedGasMassFromTransfer', data=AccretedGasMassFromTransfer)
  outf.create_dataset('AccretedGasMassFromMerger', data=AccretedGasMassFromMerger)
  outf.create_dataset('AccretedGasMassFromPristine', data=AccretedGasMassFromPristine)
  outf.create_dataset('AccretedGasMassFromPristineWind', data=AccretedGasMassFromPristineWind)

  outf.create_dataset('GasMassLost', data=GasMassLost)
  outf.create_dataset('MetalMassLost', data=MetalMassLost)
  outf.create_dataset('GasMassEjected', data=GasMassEjected)
  outf.create_dataset('StarMassFormed', data=StarMassFormed)

  outf.create_dataset('HaloID', data=HaloID)
  outf.create_dataset('Rvir', data=Rvir)
  outf.create_dataset('Mvir', data=Mvir)

  outf.close()

  print '\nDone!'



