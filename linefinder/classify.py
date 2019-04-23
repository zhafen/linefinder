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

import galaxy_dive.analyze_data.ahf as analyze_ahf
import galaxy_dive.read_data.ahf as read_ahf
import galaxy_dive.utils.astro as astro_tools
import galaxy_dive.utils.constants as constants
import galaxy_dive.utils.utilities as utilities

from .analyze_data import worldlines as analyze_worldlines

from . import config

########################################################################

default = object()

########################################################################
########################################################################


class Classifier( object ):
    '''Loads the tracked particle data, and uses that to classify the particles
    into different categories.
    '''

    @utilities.store_parameters
    def __init__(
        self,
        out_dir,
        tag,
        ptracks_tag = default,
        galids_tag = default,
        halo_data_dir = default,
        mtree_halos_index = default,
        halo_file_tag = default,
        not_in_main_gal_key = 'gal_id',
        classifications_to_save = [
            'is_unaccreted', 'is_pristine', 'is_preprocessed',
            'is_merger', 'is_mass_transfer', 'is_wind',
            'is_hitherto_EP', 'is_hitherto_NEP',
            'is_unaccreted_EP', 'is_unaccreted_NEP',
        ],
        write_events = True,
        events_to_save = [
            'is_in_other_gal', 'is_in_main_gal', 'is_accreted',
            'is_ejected', 'redshift_first_acc', 'ind_first_acc',
            'cumulative_time_in_other_gal', 'gal_event_id',
            'time_in_other_gal_before_acc_during_interval',
            'is_in_other_CGM', ],
        velocity_scale = config.VELOCITY_SCALE,
        wind_cut = config.WIND_CUT,
        absolute_wind_cut = config.ABSOLUTE_WIND_CUT,
        t_pro = config.T_PRO,
        t_m = config.T_M,
        neg = 10,
        main_halo_robustness_criteria = 'n_star',
        main_halo_robustness_value = 100,
        min_gal_density = config.GALAXY_DENSITY_CUT,
        pp_classifications_to_save = [
            'is_in_CGM', 'is_in_CGM_not_sat', 'is_in_galaxy_halo_interface',
            'is_CGM_IGM_accretion', 'is_CGM_wind',
            'is_CGM_satellite_wind', 'is_CGM_satellite_ISM',
            'CGM_fate_classifications',
        ],
    ):
        '''Setup the ID Finder.

        Args:
            out_dir (str) :
                Data directory for tracked particle data.

            tag (str) :
                Identifying tag. Currently must be put in manually.

            ptracks_tag (str, optional) :
                Identifying tag for the ptrack data. Defaults to 'tag'.

            galids_tag (str, optional) :
                Identifying tag for the galaxy_linker data. Defaults to 'tag'.

            halo_data_dir (str, optional) :
                Data directory for AHF data.
                Default value is whatever is stored in the galids file.

            mtree_halos_index (int, optional) :
                The index argument to pass to AHFReader.get_mtree_halos().
                For most cases this should be the final snapshot number, but
                see AHFReader.get_mtree_halos's documentation.
                Default value is whatever is stored in the galids file.

            halo_file_tag (str, optional) :
                What halo files to load and use? Defaults to whatever is stored
                in the galids file.

            not_in_main_gal_key (str, optional) :
                The galaxy_linker data key used to identify when not in a main
                galaxy. 'gal_id' is the default, meaning if a particle is in the
                main galaxy and isn't inside another galaxy then it's
                counted as in part of the main galaxy. Another potential option
                is 'halo_id'.

            classifications_to_save (list of strs, optional) :
                What classifications to write to a file.

            write_events (bool, optional) :
                Whether or not to save events in a particle's history to a file,
                e.g. when it's ejected.

            events_to_save (list of strs, optional) :
                What events to write to a file.

            velocity_scale (float) :
                What the velocity scale of the galaxy is (for applying velocity
                cuts).

            wind_cut (float, optional) :
                The minimum radial velocity (in units of the main galaxy
                velocity scale) a particle must have to be considered ejection.

            absolute_wind_cut (float, optional) :
                The minimum radial velocity (in km/s ) a particle must have to
                be considered ejection.

            t_pro (float, optional) :
                The processing time, i.e. the minimum time (in Myr) a particle
                must reside in a galaxy to not count as pristine gas.

            t_m (float, optional) :
                Externally-processed mass is required to spend at least t_pro
                during the interval t_m prior to accretion to qualify
                as a *merger*.

            neg (int, optional) :
                Number of earliest indices for which we neglect
                accretion/ejection events. If each indice corresponds to a
                snapshot, then it's the number of snapshots

            main_halo_robustness_criteria (str) &
            main_halo_robustness_value (int or float) :
                The main halo is considered resolved if the value of
                main_halo_robustness_criteria is greater than or equal
                to main_halo_robustness_value.
                By default the main halo is counted as resolved if the
                n_stars(main halo) >= 100.

            min_gal_density (float):
                Minimum density (n_particles in 1/cm^3) for gas to be counted as
                being in a galaxy, on top of being sufficiently close to the
                center of said galaxy.

            pp_classifications_to_save (list of strs) :
                Classifications available as part of Worldlines that will be
                saved in the classifications_*.hdf5 file.
        '''

        pass

    ########################################################################

    def classify_particles( self ):
        '''Run the full classification suite.'''

        # Print out starting information
        time_start = time.time()
        print( "#" * 80 )
        print( "Starting Classifying!" )
        print( "#" * 80 )
        print( "Using tracked particle data from this directory:\n    {}".format(
                self.out_dir
            )
        )
        print( "Using halo data from this directory:\n    {}".format(
            self.halo_data_dir
            )
        )
        print( "Data will be saved here:\n    {}".format(
                self.out_dir
            )
        )
        sys.stdout.flush()

        # Get the data files out
        self.read_data_files()

        # Do the auxiliary calculations
        print( "Calculating radial velocity, circular velocity, and dt..." )
        sys.stdout.flush()
        self.dt = self.get_time_difference()

        # Do the first wave of classifications
        print( "Identifying accretion, ejection, etc..." )
        sys.stdout.flush()
        self.is_in_other_gal = self.identify_is_in_other_gal()
        self.is_in_other_CGM = self.identify_is_in_other_CGM()
        self.is_in_main_gal = self.identify_is_in_main_gal()
        self.gal_event_id = self.calc_gal_event_id()
        self.is_accreted = self.identify_accretion()
        self.is_ejected = self.identify_ejection()

        # Information on what happens before accretion.
        print( "Figuring out what happens before first accretion..." )
        sys.stdout.flush()
        self.cum_num_acc = self.get_cum_num_acc()
        self.is_before_first_acc = self.identify_is_before_first_acc()
        self.redshift_first_acc = self.get_redshift_first_acc()
        self.cumulative_time_in_other_gal = \
            self.get_cumulative_time_in_other_gal()
        self.time_in_other_gal_before_acc = \
            self.get_time_in_other_gal_before_acc()
        self.time_in_other_gal_before_acc_during_interval = \
            self.get_time_in_other_gal_before_acc_during_interval()

        # Get the primary classifications
        print( "Performing the main classifications..." )
        sys.stdout.flush()
        self.is_hitherto_EP = self.identify_hitherto_EP()
        self.is_hitherto_NEP = self.identify_hitherto_NEP()
        self.is_unaccreted = self.identify_unaccreted()
        self.is_unaccreted_EP = self.identify_unaccreted_EP()
        self.is_unaccreted_NEP = self.identify_unaccreted_NEP()
        self.is_preprocessed = self.identify_preprocessed()
        self.is_pristine = self.identify_pristine()
        self.is_mass_transfer = self.identify_mass_transfer()
        self.is_merger = self.identify_merger()
        self.is_wind = self.identify_wind()

        # Save the results
        self.save_classifications( self.classifications_to_save )
        if self.write_events:
            self.save_events( self.events_to_save )
        self.additional_postprocessing( self.pp_classifications_to_save )

        # Print out end information
        time_end = time.time()
        print( "#" * 80 )
        print( "Done Classifying!" )
        print( "#" * 80 )
        print( "Output file saved as:\n    {}".format(
                self.classification_filepath
            )
        )
        print( "Took {:.3g} seconds, or {:.3g} seconds per particle!".format(
                time_end - time_start, (time_end - time_start) / self.n_particle
            )
        )

    ########################################################################

    def read_data_files( self ):
        '''Read the relevant data files, and store the data in a dictionary for
        easy access later on.
        '''

        print( "Reading data..." )
        sys.stdout.flush()

        self.ptrack = {}
        self.ptrack_attrs = {}

        def load_data_into_ptrack( filename, store_parameters=False ):

            filepath = os.path.join( self.out_dir, filename )
            f = h5py.File(filepath, 'r')

            # Store the particle track data in a dictionary
            for key in f.keys():
                if key != 'parameters':
                    self.ptrack[ key ] = f[ key ][...]

            # Store the ptrack attributes
            for key in f.attrs.keys():
                self.ptrack_attrs[ key ] = f.attrs[ key ]

            if store_parameters:
                default_attrs_to_replace = [
                    'halo_data_dir',
                    'mtree_halos_index',
                    'halo_file_tag',
                ]
                for attr in default_attrs_to_replace:
                    if getattr( self, attr ) is default:
                        attr_value = utilities.check_and_decode_bytes(
                            f['parameters'].attrs[attr]
                        )
                        setattr( self, attr, attr_value )
                for parameter_key in [ 'galaxy_cut', 'length_scale', ]:
                    self.ptrack_attrs[ parameter_key ] = \
                        f['parameters'].attrs[parameter_key]

            f.close()

        # Get the tag for particle tracking.
        if self.ptracks_tag is default:
            self.ptracks_tag = self.tag

        # Get the tag for galaxy finding.
        if self.galids_tag is default:
            self.galids_tag = self.tag

        # Load Particle Tracking and Galaxy Finding Data
        self.ptrack_filename = 'ptracks_{}.hdf5'.format( self.ptracks_tag )
        self.galfind_filename = 'galids_{}.hdf5'.format( self.galids_tag )
        load_data_into_ptrack( self.ptrack_filename )
        load_data_into_ptrack( self.galfind_filename, True )

        # Set useful state variables
        self.n_snap = self.ptrack['redshift'].size
        self.n_particle = self.ptrack['ID'].size

        # Get the AHF data files.
        self.ahf_reader = read_ahf.AHFReader( self.halo_data_dir )
        self.ahf_reader.get_mtree_halos( self.mtree_halos_index,
                                         self.halo_file_tag )

    ########################################################################

    def save_classifications( self, classifications_to_save ):
        '''Save the results of running the classifier.

        Args:
            classifications_to_save (list of strs) :
                What classifications to save to the file.
        '''

        # Open up the file to save the data in.
        classification_filename = 'classifications_{}.hdf5'.format( self.tag )
        self.classification_filepath = os.path.join(
            self.out_dir,
            classification_filename,
        )
        f = h5py.File( self.classification_filepath, 'a' )

        # Save the data
        for classification in classifications_to_save:

            data = getattr( self, classification )
            f.create_dataset( classification, data=data )

        utilities.save_parameters( self, f )

        # Save the current code versions
        f.attrs['linefinder_version'] = utilities.get_code_version( self )
        f.attrs['galaxy_dive_version'] = utilities.get_code_version(
            read_ahf,
            instance_type='module'
        )

        # Save the snapshot number when the main halo is first resolved.
        f.attrs['main_mt_halo_first_snap'] = self.main_mt_halo_first_snap
        f.attrs['ind_first_snap'] = self.ind_first_snap

        f.close()

    ########################################################################

    def save_events( self, events_to_save ):
        '''Save the particular events, identified during the classification process.

        Args:
            events_to_save (list of strs) : What events to save to the file.
        '''

        # Open up the file to save the data in.
        events_filename = 'events_{}.hdf5'.format( self.tag )
        self.events_filepath = os.path.join( self.out_dir, events_filename )
        f = h5py.File( self.events_filepath, 'a' )

        # Save the data
        for event_type in events_to_save:

            data = getattr( self, event_type )
            f.create_dataset( event_type, data=data )

        utilities.save_parameters( self, f )

        # Save the current code versions
        f.attrs['linefinder_version'] = utilities.get_code_version( self )
        f.attrs['galaxy_dive_version'] = utilities.get_code_version(
            read_ahf, instance_type='module' )

        f.close()

    ########################################################################

    def additional_postprocessing( self, pp_classifications_to_save ):
        '''Save additional classifications that are available as part of the
        analysis suite, but are not computed here.

        Args:
            pp_classifications_to_save (list of strs) :
                Classifications available as part of Worldlines that will be
                saved in the classifications_*.hdf5 file.
        '''

        print( "Starting final postprocessing..." )

        # Load the post-processing analysis class
        w = analyze_worldlines.Worldlines(
            data_dir = self.out_dir,
            tag = self.tag,
            ptracks_tag = self.ptracks_tag,
            galids_tag = self.galids_tag,
            halo_data_dir = self.halo_data_dir,
            mtree_halos_index = self.mtree_halos_index,
            main_halo_id = self.ptrack_attrs[ 'main_mt_halo_id' ],
            halo_file_tag = self.halo_file_tag,
        )

        # Open up the file to save the data in.
        classification_filename = 'classifications_{}.hdf5'.format( self.tag )
        self.classification_filepath = os.path.join(
            self.out_dir,
            classification_filename,
        )
        f = h5py.File( self.classification_filepath, 'a' )

        # Save the data
        for classification in pp_classifications_to_save:

            print( "Calculating {}...".format( classification ) )

            data = w.get_data( classification )
            f.create_dataset( classification, data=data )

        f.close()

    ########################################################################
    # Auxilliary Calculations
    ########################################################################

    def get_radial_velocity( self ):
        '''Get the radial velocity of particles, relative to the main galaxy.

        Returns:
            v_r ( [n_particle, n_snap] np.ndarray ) : The radial velocity of
            each particle at that redshift, relative to the main galaxy.
        '''

        # Get the position and velocity of the main galaxy
        main_mt_halo_p = self.ahf_reader.get_pos_or_vel(
            'pos', self.ptrack_attrs[ 'main_mt_halo_id' ], self.ptrack[ 'snum' ]
        )
        main_mt_halo_v = self.ahf_reader.get_pos_or_vel(
            'vel', self.ptrack_attrs[ 'main_mt_halo_id' ], self.ptrack[ 'snum' ]
        )

        # Apply cosmological corrections to the position of the main galaxy
        main_mt_halo_p *= 1. / \
            ( 1. + self.ptrack['redshift'][:, np.newaxis] ) / \
            self.ptrack_attrs['hubble']

        # Loop over each redshift
        v_r = []
        for i in range(self.n_snap):

            v = self.ptrack['V'][:, i] - main_mt_halo_v[i][np.newaxis]
            p = self.ptrack['P'][:, i] - main_mt_halo_p[i][np.newaxis]

            r_i = np.sqrt( ( p**2. ).sum( axis=1 ) )

            v_r_i = ( v * p ).sum( axis=1 )/r_i

            # Add the hubble flow.
            hubble_factor = astro_tools.hubble_parameter(
                self.ptrack['redshift'][i],
                h=self.ptrack_attrs['hubble'],
                omega_matter=self.ptrack_attrs['omega_matter'],
                omega_lambda=self.ptrack_attrs['omega_lambda'],
                units='1/s'
            )
            v_r_i += hubble_factor * r_i * constants.UNITLENGTH_IN_CM / \
                constants.UNITVELOCITY_IN_CM_PER_S

            v_r.append( v_r_i )

        # Format the output
        v_r = np.array( v_r ).transpose()

        return v_r

    ########################################################################

    def get_circular_velocity( self ):
        '''Get the circular velocity of the halo (measured at Rvir).

        Returns:
            v_c : Circular velocity of the halo in km/s, indexed the same way
            that ahf_reader.mtree_halos[i] is.
        '''

        # Get the virial radius and mass of the main galaxy
        r_vir_kpc = \
            self.ahf_reader.mtree_halos[0]['Rvir'][ self.ptrack[ 'snum' ] ]
        m_vir_msun = \
            self.ahf_reader.mtree_halos[0]['Mvir'][ self.ptrack[ 'snum' ] ]

        # Convert r_vir and m_vir to physical units
        r_vir_kpc *= \
            1. / ( 1. + self.ptrack['redshift'] ) / self.ptrack_attrs['hubble']
        m_vir_msun /= self.ptrack_attrs['hubble']

        v_c = astro_tools.circular_velocity( r_vir_kpc, m_vir_msun )

        return v_c

    ########################################################################

    def get_velocity_scale( self ):
        '''Get the characteristic velocity scale.

        Returns:
            velocity_scale (np.ndarray): Velocity of the halo in proper km/s.
        '''

        if self.velocity_scale == 'Vc(Rvir)':
            return self.get_circular_velocity()
        elif self.velocity_scale == 'Vc(Rgal)':
            ahf_key_parser = analyze_ahf.HaloKeyParser()
            r_gal_key = ahf_key_parser.get_velocity_at_radius_key(
                'Vc',
                self.ptrack_attrs['galaxy_cut'],
                self.ptrack_attrs['length_scale'],
            )
            main_mt_halo_id = self.ptrack_attrs[ 'main_mt_halo_id' ]
            mtree_halo = self.ahf_reader.mtree_halos[main_mt_halo_id]
            return mtree_halo[r_gal_key][ self.ptrack[ 'snum' ] ]
        else:
            main_mt_halo_id = self.ptrack_attrs[ 'main_mt_halo_id' ]
            mtree_halo = self.ahf_reader.mtree_halos[main_mt_halo_id]
            return mtree_halo[self.velocity_scale][ self.ptrack[ 'snum' ] ]

    ########################################################################

    def get_time_difference( self ):
        '''Get the time between snapshots.

        Returns:
            dt ([n_particle, n_snap-1] np.array): Time between snapshots in Myr.
        '''

        # Replicate redshifts self.ptrack indexing (last one removed)
        redshift = np.tile(
            self.ptrack['redshift'][0:self.n_snap], (self.n_particle, 1) )

        # Age of the universe in Myr
        time = astro_tools.age_of_universe(
            redshift,
            h=self.ptrack_attrs['hubble'],
            omega_matter=self.ptrack_attrs['omega_matter']
        )
        dt = time[:, :-1] - time[:, 1:]

        return dt

    ########################################################################
    # Auxilliary Classification Methods
    ########################################################################

    def identify_is_in_other_gal( self ):
        '''Identify what particles are in a galaxy besides the main galaxy.

        Returns:
            is_in_other_gal ( [n_particle, n_snap-1] np.ndarray of bools) :
                True if in a galaxy other than the main galaxy at
                that redshift.
        '''

        # Get the ID of the main halo for a given snapshot
        # (remember that the mtree halo ID isn't the same as the ID at a given
        # snapshot).
        main_mt_halo_id = self.ptrack_attrs['main_mt_halo_id']
        main_mtree_halo = self.ahf_reader.mtree_halos[ main_mt_halo_id ]
        main_halo_id = main_mtree_halo[ 'ID' ][ self.ptrack[ 'snum' ] ]
        main_halo_id_tiled = np.tile( main_halo_id, ( self.n_particle, 1 ) )

        # Check if we're inside the galaxy/halo other than the main galaxy
        # This step is necessary, and the inverse of it is not redundant,
        # because it removes anything that's in the
        # main halo *and* that's the least massive galaxy it's in.
        is_not_in_main_gal = ( self.ptrack[self.not_in_main_gal_key] !=
                               main_halo_id_tiled )
        is_in_gal = ( self.ptrack['gal_id'] >= 0 )

        is_in_other_gal = ( is_in_gal & is_not_in_main_gal )

        # If there's a density requirement, apply it.
        if self.min_gal_density is not None:
            is_in_other_gal = ( is_in_other_gal &
                                self.meets_density_requirement )

        return is_in_other_gal

    ########################################################################

    def identify_is_in_other_CGM( self ):
        '''Identify what particles are in a CGM besides the main CGM.

        Returns:
            is_in_other_cgm ( [n_particle, n_snap-1] np.ndarray of bools) :
                True if in a CGM other than the main CGM at
                that redshift.
        '''

        # Get the ID of the main halo for a given snapshot
        # (remember that the mtree halo ID isn't the same as the ID at a given
        # snapshot).
        main_mt_halo_id = self.ptrack_attrs['main_mt_halo_id']
        main_mtree_halo = self.ahf_reader.mtree_halos[ main_mt_halo_id ]
        main_halo_id = main_mtree_halo[ 'ID' ][ self.ptrack[ 'snum' ] ]
        main_halo_id_tiled = np.tile( main_halo_id, ( self.n_particle, 1 ) )

        # Check if we're inside the galaxy/halo other than the main CGM
        # This step is necessary, and the inverse of it is not redundant,
        # because it removes anything that's in the
        # main CGM *and* that's the least massive halo it's in.
        is_not_only_in_main_cgm = (
            self.ptrack['1.0_Rvir'] != main_halo_id_tiled
        )
        is_in_cgm = ( self.ptrack['1.0_Rvir'] >= 0 )

        is_in_other_cgm = ( is_in_cgm & is_not_only_in_main_cgm )

        return is_in_other_cgm

    ########################################################################

    def identify_is_in_main_gal( self ):
        '''Identify what particles are in a main galaxy. They must be in the
        main galaxy *and* not inside any other galaxy
        at that redshift, even a subhalo galaxy.

        Returns:
            is_in_main_gal ( [n_particle, n_snap-1] np.ndarray of bools) :
                True if in the main galaxy
                (and not any other galaxy) at that redshift.
        '''

        is_not_in_other_gal = np.invert( self.is_in_other_gal )

        # If we're literally inside the main galaxy
        ptrack_mt_gal_id = self.ptrack['mt_gal_id'][:, 0:self.n_snap]
        main_mt_halo_id = self.ptrack_attrs['main_mt_halo_id']
        is_in_main_gal_literal = ( ptrack_mt_gal_id == main_mt_halo_id )

        # Find if particles are inside/outside of main galaxy at each redshift
        is_in_main_gal = ( is_in_main_gal_literal & is_not_in_other_gal )

        # Correct for boundary conditions
        main_gal_not_resolved = ( self.ptrack['snum'] <
                                  self.main_mt_halo_first_snap )
        main_gal_not_resolved_inds = np.where( main_gal_not_resolved )[0]
        is_in_main_gal[slice(None), main_gal_not_resolved_inds] = False

        # If there's a density requirement, apply it.
        if self.min_gal_density is not None:
            is_in_main_gal = ( is_in_main_gal & self.meets_density_requirement )

        return is_in_main_gal

    ########################################################################

    def calc_gal_event_id( self ):
        '''Identify potential accretion/ejection events relative to main galaxy
        at each redshift

        Returns:
            gal_event_id ( [n_particle, n_snap-1] np.ndarray of ints) :
                GalEvent = 0 (no change), 1 (entering galaxy),
                -1 (leaving galaxy) at that redshift
        '''

        # Find when the particles enter and exit the galaxy
        is_in_main_gal_after = \
            self.is_in_main_gal[:, 0:self.n_snap - 1].astype( int )
        is_in_main_gal_before = \
            self.is_in_main_gal[:, 1:self.n_snap].astype( int )
        gal_event_id = is_in_main_gal_after - is_in_main_gal_before

        return gal_event_id

    ########################################################################

    def identify_accretion( self ):
        '''Identify ALL gas/star accretion events, i.e. whether or not a
        particle was outside the galaxy at one redshift,
        and inside at the next'''

        is_accreted = ( self.gal_event_id == 1 )

        return is_accreted

    ########################################################################

    def identify_ejection( self ):
        '''Identify ALL gas wind ejection events.
            These conditions must be met to identify as ejection:
                1. Inside the main galaxy at one snapshot, and not at the
                   previous snapshot.
                2. Radial velocity of the particle relative to the main galaxy
                    must be greater than some fraction of the
                    circular velocity of the main galaxy.
                3. Radial velocity of the particle relative to the main galaxy
                    must be greater than some base speed.
                4. The particle must be a gas particle.
                5. The particle must be outside any other galaxy.
        '''

        # Get the radial velocity out.
        v_r = self.get_radial_velocity()

        # Get the circular velocity out and tile it for comparison
        v_scale = self.get_velocity_scale()
        v_scale_tiled = np.tile( v_scale, ( self.n_particle, 1 ) )

        # The conditions for being outside any galaxy
        # Condition 1
        is_outside_before_inside_after = ( self.gal_event_id == -1 )
        # Condition 2
        has_minimum_vr_in_vc = ( v_r[:, 0:self.n_snap - 1] >
                                 self.wind_cut *
                                 v_scale_tiled[:, 0:self.n_snap - 1] )
        # Condition 3
        has_minimum_vr = ( v_r[:, 0:self.n_snap - 1] > self.absolute_wind_cut )
        # Condition 4
        is_gas = ( self.ptrack['PType'][:, 0:self.n_snap - 1] == 0 )
        # Condition 5
        is_not_in_other_gal = np.invert( self.is_in_other_gal[:, 0:self.n_snap - 1] )

        is_ejected = (
            is_outside_before_inside_after &
            has_minimum_vr_in_vc &
            has_minimum_vr &
            is_gas &
            is_not_in_other_gal
        )

        return is_ejected

    ########################################################################
    # What happens before accretion?
    ########################################################################

    def get_cum_num_acc( self ):
        '''Get the cumulative number of accretions so far.

        Returns:
            cum_num_acc ([n_particle, n_snap-1] np.ndarray of ints) :
                Cumulative number of accretion events for that particles.
        '''

        # Index to revert order of redshift snapshots
        ind_rev = np.arange( self.n_snap - 2, -1, -1 )

        # cumulative number of accretion events
        reverse_cum_num_acc = self.is_accreted[:, ind_rev].cumsum(axis=1)
        cum_num_acc = reverse_cum_num_acc[:, ind_rev]

        return cum_num_acc

    ########################################################################

    def identify_is_before_first_acc( self ):
        '''Identify when before a particle's first accretion event.

        Returns:
            is_before_first_acc ([n_particle, n_snap-1] np.ndarray of bools) :
                If True, then the first accretion event for that particle
                hasn't happened yet.
        '''
        is_before_first_acc = ( self.cum_num_acc == 0 ) & \
            ( self.is_in_main_gal[:, 0:self.n_snap - 1] == 0 )

        return is_before_first_acc

    ########################################################################

    @property
    def ind_first_acc( self ):
        '''Get the index of first accretion.
        This is defined as the the indice immediately after accretion happens.

        Returns:
            ind_first_acc ([n_particle,] np.ndarray of floats):
                Index of first accretion.
        '''

        if not hasattr( self, '_ind_first_acc' ):
            inds = np.arange( self.ptrack['redshift'].size )
            inds_tiled_full = np.tile( inds, ( self.n_particle, 1 ) )
            inds_tiled = inds_tiled_full[:, 0:self.n_snap - 1]

            self._ind_first_acc = np.ma.masked_array(
                inds_tiled, mask=self.is_before_first_acc ).max( axis=1 )
            self._ind_first_acc = self._ind_first_acc.filled(
                fill_value = config.INT_FILL_VALUE )

            # Mask the ones that were always part of the galaxy
            always_part_of_gal = self.is_before_first_acc.sum( axis=1 ) == 0
            self._ind_first_acc[always_part_of_gal] = config.INT_FILL_VALUE

        return self._ind_first_acc

    ########################################################################

    def get_redshift_first_acc( self ):
        '''Get the redshift of first accretion.

        Returns:
            redshift_first_acc ([n_particle,] np.ndarray of floats):
                Redshift of first accretion.
        '''

        redshift_tiled_full = np.tile(
            self.ptrack['redshift'], ( self.n_particle, 1 ) )
        redshift_tiled = redshift_tiled_full[:, 0:self.n_snap - 1]

        redshift_first_acc = np.ma.masked_array(
            redshift_tiled, mask=self.is_before_first_acc ).max( axis=1 )
        redshift_first_acc = redshift_first_acc.filled( fill_value = -1. )

        # Mask the ones that were always part of the galaxy
        always_part_of_gal = self.is_before_first_acc.sum( axis=1 ) == 0
        redshift_first_acc[always_part_of_gal] = -1.

        return redshift_first_acc

    ########################################################################

    def get_cumulative_time_in_other_gal( self ):
        '''Get the amount of time in galaxies besides the main galaxy before
        being accreted.

        For a single time in another galaxy, this is the
        ( age of universe at the last snapshot before the conditions are true )
        - ( age of the universe at the last snapshot where the conditions
        are true ), and generalizes to multiple events in other galaxies.

        Returns:
            cumulative_time_in_other_gal ([ n_particle, n_snap ]
                np.ndarray of floats):
                Time in another galaxy at a given snapshot.
        '''
        other_gal = self.is_in_other_gal[:, 0:self.n_snap - 1].astype( float )
        dt_in_other_gal = ( self.dt * other_gal )

        dt_in_other_gal_reversed = np.fliplr( dt_in_other_gal )

        cumulative_time_in_other_gal_reversed = \
            dt_in_other_gal_reversed.cumsum( axis=1 )

        cumulative_time_in_other_gal = np.fliplr(
            cumulative_time_in_other_gal_reversed )

        return cumulative_time_in_other_gal

    ########################################################################

    def get_time_in_other_gal_before_acc( self ):
        '''Get the amount of time in galaxies besides the main galaxy before
        being accreted.
        For a single time in another galaxy, this is the
        ( age of universe at the last snapshot before the conditions are true )
        - ( age of the universe at the last snapshot where the conditions
        are true ), and generalizes to multiple events in other galaxies.

        Returns:
            time_in_other_gal_before_acc ([ n_particle, ] np.ndarray of floats):
                Time in another galaxy before being first accreted onto the
                main galaxy.
        '''

        is_in_other_gal_before_first_acc = (
            self.is_before_first_acc &
            self.is_in_other_gal[:, 0:self.n_snap - 1] )
        time_in_other_gal_before_acc = (
            self.dt * is_in_other_gal_before_first_acc.astype( float )
        ).sum(axis=1)

        return time_in_other_gal_before_acc

    ########################################################################

    def get_time_in_other_gal_before_acc_during_interval( self ):
        '''Get the amount of time in galaxies besides the main galaxy before
        being accreted, during an interval before being accreted.

        Returns:
            time_in_other_gal_before_acc_during_interval
            ([ n_particle, ] np.ndarray of floats) :
                Time in another galaxy before being first accreted onto the
                main galaxy, within some recent time interval
        '''

        # Get the total amount of time before being accreted
        cum_time_before_acc = (
            self.dt * self.is_before_first_acc.astype( float )
        ).cumsum(axis=1)

        # Conditions for counting up time
        time_interval = self.t_m
        is_in_other_gal_in_time_interval_before_acc = (
            # Count up only the time before first accretion.
            ( cum_time_before_acc <= time_interval ) &
            # Make sure we haven't accreted yet
            self.is_before_first_acc &
            # Make sure we're in another galaxy at that time
            self.is_in_other_gal[:, 0:self.n_snap - 1]
        )

        time_in_other_gal_before_acc_during_interval = (
            self.dt *
            is_in_other_gal_in_time_interval_before_acc.astype( float )
        ).sum(axis=1)

        return time_in_other_gal_before_acc_during_interval

    ########################################################################
    # Main Classification Methods
    ########################################################################

    def identify_hitherto_EP( self ):
        '''Identify particles that have been processed by another galaxy by
        the tabulated snapshot.

        Returns:
            is_hitherto_EP ( [n_particle,n_snap] np.ndarray of bools ) :
                True for particle i at snapshot j if it has spent at least
                t_pro in another galaxy by that point.
        '''

        is_hitherto_EP = self.cumulative_time_in_other_gal > self.t_pro

        # Correct for length of is_EP (since self.cumulative_time_in_other_gal
        # doesn't extend to all snapshots)
        values_to_append = np.array( [ False, ] * self.n_particle )
        is_hitherto_EP = np.insert(
            is_hitherto_EP, -1, values_to_append, axis=1 )

        return is_hitherto_EP

    ########################################################################

    def identify_hitherto_NEP( self ):
        '''Identify particles that have not been processed by another galaxy by
        the tabulated snapshot.

        Returns:
            is_hitherto_EP ( [n_particle,n_snap] np.ndarray of bools ) :
                True for particle i at snapshot j if it has not spent at least
                t_pro in another galaxy by that point.
        '''

        is_hitherto_NEP = self.cumulative_time_in_other_gal <= self.t_pro

        # Correct for length of is_NEP (since self.cumulative_time_in_other_gal
        # doesn't extend to all snapshots)
        values_to_append = np.array( [ True, ] * self.n_particle )
        is_hitherto_NEP = np.insert(
            is_hitherto_NEP, -1, values_to_append, axis=1 )

        return is_hitherto_NEP

    ########################################################################

    def identify_unaccreted( self ):
        '''Identify particles never accreted onto the main galaxy.

        Returns:
            is_unaccreted ( [n_particle,] np.ndarray of bools ) :
                True for particle i if it has never been inside the main galaxy.
        '''

        n_snaps_in_gal = self.is_in_main_gal.sum( axis=1 )

        is_unaccreted = n_snaps_in_gal == 0

        return is_unaccreted

    ########################################################################

    def identify_unaccreted_EP( self ):
        '''Identify particles never accreted onto the main galaxy that have
        spent at least t_pro in another galaxy by the specified snapshot.

        Returns:
            is_unaccreted_EP ( [n_particle,n_snap] np.ndarray of bools ) :
                True for particle i at snapshot j if it has spent at least
                t_pro in another galaxy by that point and never accretes onto
                the main galaxy.
        '''

        unaccreted_tiled_rot = np.tile( self.is_unaccreted, ( self.n_snap, 1) )
        unaccreted_tiled = unaccreted_tiled_rot.transpose()

        is_unaccreted_EP = unaccreted_tiled & self.is_hitherto_EP

        return is_unaccreted_EP

    ########################################################################

    def identify_unaccreted_NEP( self ):
        '''Identify particles never accreted onto the main galaxy that have not
        spent at least t_pro in another galaxy by the specified snapshot.

        Returns:
            is_unaccreted_NEP ( [n_particle,n_snap] np.ndarray of bools ) :
                True for particle i at snapshot j if it has not spent at least
                t_pro in another galaxy by that point.
        '''

        unaccreted_tiled_rot = np.tile( self.is_unaccreted, ( self.n_snap, 1) )
        unaccreted_tiled = unaccreted_tiled_rot.transpose()

        is_unaccreted_NEP = unaccreted_tiled & self.is_hitherto_NEP

        return is_unaccreted_NEP

    ########################################################################

    def identify_preprocessed( self ):
        '''Identify pre-proceesed gas, or "externally processed" gas.

        Returns:
            is_preprocessed ( [n_particle] np.ndarray of bools ) :
                True for particle i if it has spent at least some minimum
                amount of time in another galaxy before being accreted.
        '''

        is_preprocessed = ( self.time_in_other_gal_before_acc > self.t_pro )

        # Apply "boundary conditions": particles inside galaxy when it's first
        # resolved count as pristine
        bc_should_be_applied = ( self.ind_first_acc >
                                 self.ind_first_snap - self.neg )
        is_preprocessed[bc_should_be_applied] = False

        # Make sure that every particle classified as unaccreted is not also
        # classified as preprocessed
        is_preprocessed[self.is_unaccreted] = False

        return is_preprocessed

    ########################################################################

    def identify_pristine( self ):
        '''Identify pristine gas, or "non-externally processed" gas.

        Returns:
            is_pristine ( [n_particle] np.ndarray of bools ) :
                True for particle i if it has never spent some minimum amount
                of time in another galaxy before being accreted.
        '''

        # Anything that's not preprocessed or unaccreted is pristine, by
        # definition
        is_preprocessed_or_unaccreted = np.ma.mask_or(
            self.is_preprocessed, self.is_unaccreted )
        is_pristine = np.invert( is_preprocessed_or_unaccreted )

        return is_pristine

    ########################################################################

    def identify_mass_transfer( self ):
        '''Boolean for whether or no particles are from mass transfer

        Returns:
            is_mass_transfer (np.ndarray of bools) :
                True for particle i if it has been preprocessed but has *not*
                spent at least some minimum amount of time in another galaxy in
                a recent interval.
        '''

        has_not_spent_minimum_time = (
            self.time_in_other_gal_before_acc_during_interval < self.t_pro )
        is_mass_transfer = ( self.is_preprocessed & has_not_spent_minimum_time )

        return is_mass_transfer

    ########################################################################

    def identify_merger( self ):
        '''Boolean for whether or no particles are from galaxies merging.

        Returns:
            is_merger ( [n_particle] np.ndarray of bools ) :
                True for particle i if it has been preprocessed and has
                spent at least some minimum amount of time in another galaxy in
                a recent interval.
        '''
        has_spent_minimum_time = (
            self.time_in_other_gal_before_acc_during_interval >= self.t_pro )
        is_merger = (  self.is_preprocessed & has_spent_minimum_time  )

        return is_merger

    ########################################################################

    def identify_wind( self ):
        '''Boolean for whether or not particles are from wind.

        Returns:
            is_wind ( [n_particle] np.ndarray of bools ) :
                True for particle i if it has been ejected at least once before
                snapshot n
        '''

        # Index to revert order of redshift snapshots
        ind_rev = np.arange( self.n_snap - 2, -1, -1 )

        # Cumulative number of ejection events
        cum_num_eject = self.is_ejected[:, ind_rev].cumsum( axis=1 )[:, ind_rev]

        # Set up and build is_wind
        is_wind = np.zeros( ( self.n_particle, self.n_snap ), dtype=np.int32 )
        is_wind[:, 0:self.n_snap - 1] = ( cum_num_eject >= 1 )

        return is_wind.astype( bool )

    ########################################################################
    # Properties
    ########################################################################

    @property
    def main_mt_halo_first_snap( self ):
        '''Find the first snapshot at which the main merger tree halo is
        resolved.
        '''

        if not hasattr( self, '_main_mt_halo_first_snap' ):

            main_mt_halo_id = self.ptrack_attrs['main_mt_halo_id']
            mtree_halo = self.ahf_reader.mtree_halos[main_mt_halo_id]

            snapshot = np.argmax(
                mtree_halo[self.main_halo_robustness_criteria][::-1] >=
                self.main_halo_robustness_value )

            self._main_mt_halo_first_snap = snapshot

        return self._main_mt_halo_first_snap

    ########################################################################

    @property
    def ind_first_snap( self ):
        '''Find the indice for first snapshot at which the main merger tree halo
        is resolved.
        '''

        if not hasattr( self, '_ind_first_snap' ):

            # In the case that we aren't tracking over the full range of data,
            # and our first tracked snapshot comes after the first snapshot at
            # which the merger tree is resolved,
            # we set the first indice at which the main merger tree halo is
            # resolved to the last indice in our array.
            if self.main_mt_halo_first_snap < self.ptrack['snum'].min():
                self._ind_first_snap = -1

            else:

                # Look for the first stored snapshot above
                # self.main_mt_halo_first_snap
                search_snap = self.main_mt_halo_first_snap
                search_for_first_ind = True
                while search_for_first_ind:
                    potential_inds = np.where(
                        self.ptrack['snum'] == search_snap )[0]

                    # We found a viable index
                    if potential_inds.size == 1:
                        search_for_first_ind = False
                    # Throw an exception if we go too far
                    elif search_snap > self.ptrack['snum'].max():
                        raise Exception( "Found no viable first index." )
                    else:
                        search_snap += 1

                self._ind_first_snap = potential_inds[0]

        return self._ind_first_snap

    ########################################################################

    @property
    def meets_density_requirement( self ):
        '''Find particles that are either stars or have sufficient density to
        be counted as part of a galaxy.
        '''

        if not hasattr( self, '_meets_density_requirement' ):
            is_gas = self.ptrack['PType'] == config.PTYPE_GAS
            is_star = self.ptrack['PType'] == config.PTYPE_STAR
            has_minimum_density = self.ptrack['Den'] > self.min_gal_density

            is_gas_and_meets_density_requirement = (
                is_gas & has_minimum_density )
            self._meets_density_requirement = np.ma.mask_or(
                is_star, is_gas_and_meets_density_requirement )

        return self._meets_density_requirement
