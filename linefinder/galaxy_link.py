#!/usr/bin/env python
'''Means to associate particles with galaxies and halos at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import gc
import h5py
import jug
import numpy as np
import os
import sys
import time

import galaxy_dive.analyze_data.halo_data as halo_data
import galaxy_dive.galaxy_linker.linker as galaxy_linker
import galaxy_dive.utils.mp_utils as mp_utils
import galaxy_dive.utils.utilities as utilities

from . import config

########################################################################

default = object()

########################################################################
########################################################################


class ParticleTrackGalaxyLinker( object ):
    '''Find the association with galaxies for entire particle tracks.'''

    @utilities.store_parameters
    def __init__(
        self,
        out_dir,
        tag,
        main_mt_halo_id,
        mtree_halos_index = None,
        halo_file_tag = 'smooth',
        halo_data_dir = default,
        ptracks_tag = default,
        galaxy_cut = config.GALAXY_CUT,
        length_scale = config.LENGTH_SCALE,
        mt_length_scale = config.MT_LENGTH_SCALE,
        ids_to_return = [
            'gal_id',
            'mt_gal_id',
            '1.0_Rvir',
            '2.0_Rvir',
            'd_other_gal',
            'd_other_gal_scaled',
        ],
        ids_with_supplementary_data = [],
        supplementary_data_keys = [],
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
                What is the ID of the main halo. To automatically choose via the
                most massive at z=0, set equal to None. HOWEVER, be warned that
                the most massive halo at z=0 *may not* be the main halo.

            mtree_halos_index (str or int) :
                The index argument to pass to HaloData.get_mtree_halos().
                For most cases this should be the final snapshot number, but see
                HaloData.get_mtree_halos's documentation.

            halo_file_tag (str) :
                What halo files to load (e.g. defaults to loading
                `halo_00000_smooth.dat`, etc.)?

            halo_data_dir (str, optional) :
                Directory the halo data is in. Defaults to the directory the
                simulation data is stored in, as found in the ptracks file.

            ptracks_tag (str, optional) :
                Identifying tag for the ptrack data. Defaults to tag.

            galaxy_cut (float, optional) :
                Anything within galaxy_cut*length_scale is counted as being
                inside the galaxy

            length_scale (str, optional) :
                Anything within galaxy_cut*length_scale is counted as being
                inside the galaxy.

            mt_length_scale (str, optional) :
                Same as length_scale, but for merger tree galaxies

            ids_to_return (list of strs, optional):
                The types of id you want to get out.

            ids_with_supplementary_data (list of strs, optional) :
                What types of IDs should include supplementary data pulled
                from the halo files.

            supplementary_data_keys (list of strs, optional) :
                What data keys in the halo files should be accessed and
                included as part of supplementary data.

            minimum_criteria (str, optional) :
                Options...
                'n_star' -- halos must contain a minimum number of stars to
                    count as containing a galaxy.
                'M_star' -- halos must contain a minimum stellar mass to count
                    as containing a galaxy.

            minimum_value (int or float, optional) :
                The minimum amount of something (specified in minimum criteria)
                in order for a halo to count as hosting a galaxy.

            n_processors (int) :
                The number of processors to use. If parallel, expect significant
                memory usage.
        '''

        pass

    ########################################################################

    def find_galaxies_for_particle_tracks( self ):
        '''Main function.'''

        self.time_start = time.time()

        print( "#" * 80 )
        print( "Starting Adding Galaxy and Halo IDs!" )
        print( "#" * 80 )
        print( "Using halo data from this directory:\n    {}".format(
                self.halo_data_dir
            )
        )
        print( "Data will be saved here:\n    {}".format( self.out_dir ) )
        sys.stdout.flush()

        self.read_data()

        if self.n_processors > 1:
            self.get_galaxy_identification_loop_parallel()
        else:
            self.get_galaxy_identification_loop()

        self.write_galaxy_identifications( self.ptrack_gal_ids )

        time_end = time.time()

        print( "#" * 80 )
        print( "Done Adding Galaxy and Halo IDs!" )
        print( "#" * 80 )
        print( "The data was saved at:\n    {}".format( self.save_filepath ) )
        print( "Took {:.3g} seconds, or {:.3g} seconds per particle!".format(
                time_end - self.time_start,
                (time_end - self.time_start) / self.n_particles
            )
        )

    ########################################################################

    def find_galaxies_for_particle_tracks_jug( self ):
        '''Main function when using jug'''

        self.read_data()

        ptrack_gal_ids = self.get_galaxy_identification_loop_jug()

        jug.Task( self.write_galaxy_identifications, ptrack_gal_ids )

        jug.barrier()

    ########################################################################

    def read_data( self ):
        '''Read the input data.

        Modifies:
            self.ptrack (h5py file) : Loaded tracked particle data.
            self.halo_data (HaloData instance): For the halo data.
        '''

        # Get the tag for particle tracking.
        if self.ptracks_tag is default:
            self.ptracks_tag = self.tag

        # Load the particle track data
        ptrack_filename = 'ptracks_{}.hdf5'.format( self.ptracks_tag )
        self.ptrack_filepath = os.path.join( self.out_dir, ptrack_filename )
        self.ptrack = h5py.File( self.ptrack_filepath, 'r' )

        if self.halo_data_dir is default:
            self.halo_data_dir = self.ptrack['parameters'].attrs['sdir']

        # Load the halo data
        self.halo_data = halo_data.HaloData( self.halo_data_dir )

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

            # Get the data parameters to pass to GalaxyLinker
            kwargs = {
                'halo_data': self.halo_data,
                'galaxy_cut': self.galaxy_cut,
                'length_scale': self.length_scale,
                'mt_length_scale': self.mt_length_scale,
                'ids_to_return': self.ids_to_return,
                'ids_with_supplementary_data': self.ids_with_supplementary_data,
                'supplementary_data_keys': self.supplementary_data_keys,
                'minimum_criteria': self.minimum_criteria,
                'minimum_value': self.minimum_value,

                'redshift': self.ptrack['redshift'][...][ i ],
                'snum': self.ptrack['snum'][...][ i ],
                'hubble': self.ptrack.attrs['hubble'],
                'halo_data_dir': self.halo_data_dir,
                'mtree_halos_index': self.mtree_halos_index,
                'main_mt_halo_id': self.main_mt_halo_id,
                'halo_file_tag': self.halo_file_tag,
            }

            time_start = time.time()

            # Find the galaxy for a given snapshot
            gal_linker = galaxy_linker.GalaxyLinker(
                particle_positions, **kwargs )
            galaxy_and_halo_ids = gal_linker.find_ids()

            time_end = time.time()

            print( 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'\
                .format(
                    kwargs['snum'],
                    kwargs['redshift'],
                    time_end - time_start
                )
            )
            sys.stdout.flush()

            # Make the arrays to store the data in
            if not hasattr( self, 'ptrack_gal_ids' ):
                self.ptrack_gal_ids = {}
                for key in galaxy_and_halo_ids.keys():
                    dtype = type( galaxy_and_halo_ids[key][0] )
                    self.ptrack_gal_ids[key] = np.empty(
                        ( gal_linker.n_particles, n_snaps ), dtype=dtype )

            # Store the data in the primary array
            for key in galaxy_and_halo_ids.keys():
                self.ptrack_gal_ids[key][ :, i ] = galaxy_and_halo_ids[key]

            # Try clearing up memory again, in case gal_linker is hanging around
            del kwargs
            del gal_linker
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
            gal_linker = galaxy_linker.GalaxyLinker(
                particle_positions, **kwargs )
            galaxy_and_halo_ids = gal_linker.find_ids()

            time_end = time.time()

            print( 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'\
                .format(
                    kwargs['snum'],
                    kwargs['redshift'],
                    time_end - time_start
                )
            )
            sys.stdout.flush()

            # Try to avoid memory leaks
            del kwargs
            del gal_linker
            gc.collect()

            return galaxy_and_halo_ids

        n_snaps = self.ptrack['snum'][...].size
        n_particles = self.ptrack['P'][...].shape[0]

        # Loop over each included snapshot to get args
        all_args = []
        for i in range( n_snaps ):

            # Get the particle positions
            particle_positions = self.ptrack['P'][...][ :, i ]

            # Get the data parameters to pass to GalaxyLinker
            kwargs = {
                'halo_data': None,
                'galaxy_cut': self.galaxy_cut,
                'length_scale': self.length_scale,
                'mt_length_scale': self.mt_length_scale,
                'ids_to_return': self.ids_to_return,
                'minimum_criteria': self.minimum_criteria,
                'minimum_value': self.minimum_value,

                'redshift': self.ptrack['redshift'][...][ i ],
                'snum': self.ptrack['snum'][...][ i ],
                'hubble': self.ptrack.attrs['hubble'],
                'halo_data_dir': self.halo_data_dir,
                'mtree_halos_index': self.mtree_halos_index,
                'main_mt_halo_id': self.main_mt_halo_id,
                'halo_file_tag': self.halo_file_tag,
            }

            all_args.append( (particle_positions, kwargs) )

        # Actual parallel calculation
        galaxy_and_halo_ids_all = mp_utils.parmap(
            get_galaxy_and_halo_ids, all_args, self.n_processors )

        assert len( galaxy_and_halo_ids_all ) == n_snaps

        # Store the results
        for i, galaxy_and_halo_ids in enumerate( galaxy_and_halo_ids_all ):

            # Make the arrays to store the data in
            if not hasattr( self, 'ptrack_gal_ids' ):
                self.ptrack_gal_ids = {}
                for key in galaxy_and_halo_ids.keys():
                    dtype = type( galaxy_and_halo_ids[key][0] )
                    self.ptrack_gal_ids[key] = np.empty(
                        ( n_particles, n_snaps ), dtype=dtype )

            # Store the data in the primary array
            for key in galaxy_and_halo_ids.keys():
                self.ptrack_gal_ids[key][ :, i ] = galaxy_and_halo_ids[key]

            # Try clearing up memory again, in case gal_linker is hanging around
            del galaxy_and_halo_ids
            gc.collect()

    ########################################################################

    def get_galaxy_identification_loop_jug( self ):
        '''Loop over all snapshots and identify the galaxy in each.
        Use Jug for parallelism.

        Modifies:
            self.ptrack_gal_ids (dict) : Where the galaxy IDs are stored.
        '''

        def get_galaxy_and_halo_ids( i ):
            '''Get the galaxy and halo ids for a single snapshot.'''

            # Get the particle positions
            particle_positions = self.ptrack['P'][...][ :, i ]

            # Get the data parameters to pass to GalaxyLinker
            kwargs = {
                'halo_data': None,
                'galaxy_cut': self.galaxy_cut,
                'length_scale': self.length_scale,
                'mt_length_scale': self.mt_length_scale,
                'ids_to_return': self.ids_to_return,
                'minimum_criteria': self.minimum_criteria,
                'minimum_value': self.minimum_value,

                'redshift': self.ptrack['redshift'][...][ i ],
                'snum': self.ptrack['snum'][...][ i ],
                'hubble': self.ptrack.attrs['hubble'],
                'halo_data_dir': self.halo_data_dir,
                'mtree_halos_index': self.mtree_halos_index,
                'main_mt_halo_id': self.main_mt_halo_id,
                'halo_file_tag': self.halo_file_tag,
            }

            time_start = time.time()

            # Find the galaxy for a given snapshot
            gal_linker = galaxy_linker.GalaxyLinker(
                particle_positions, **kwargs )
            galaxy_and_halo_ids = gal_linker.find_ids()

            time_end = time.time()

            print( 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'\
                .format(
                    kwargs['snum'],
                    kwargs['redshift'],
                    time_end - time_start
                )
            )
            sys.stdout.flush()

            # Try to avoid memory leaks
            del kwargs
            del gal_linker
            gc.collect()

            return galaxy_and_halo_ids

        n_snaps = self.ptrack['snum'][...].size
        n_particles = self.ptrack['P'][...].shape[0]

        # Loop over each included snapshot and submit Jug Tasks
        galaxy_and_halo_ids_all = []
        for i in range( n_snaps ):

            galaxy_and_halo_ids = jug.Task(
                get_galaxy_and_halo_ids,
                i,
            )

            galaxy_and_halo_ids_all.append( galaxy_and_halo_ids )

        assert len( galaxy_and_halo_ids_all ) == n_snaps

        # Store the results
        def store_results( galaxy_and_halo_ids_all ):
            for i, galaxy_and_halo_ids in enumerate( galaxy_and_halo_ids_all ):

                # Make the arrays to store the data in
                if not hasattr( self, 'ptrack_gal_ids' ):
                    self.ptrack_gal_ids = {}
                    for key in galaxy_and_halo_ids.keys():
                        dtype = type( galaxy_and_halo_ids[key][0] )
                        self.ptrack_gal_ids[key] = np.empty(
                            ( n_particles, n_snaps ), dtype=dtype )

                # Store the data in the primary array
                for key in galaxy_and_halo_ids.keys():
                    self.ptrack_gal_ids[key][ :, i ] = galaxy_and_halo_ids[key]

                # Try clearing up memory again, in case gal_linker
                # is hanging around
                del galaxy_and_halo_ids
                gc.collect()

            return self.ptrack_gal_ids

        return jug.Task( store_results, galaxy_and_halo_ids_all )

    ########################################################################

    def write_galaxy_identifications( self, ptrack_gal_ids ):
        '''Write the data, close the file, and print out information.'''

        # Get the number of particles, for use in reporting the time
        self.n_particles = self.ptrack[ 'Den' ][...].shape[0]

        # Close the old dataset
        self.ptrack.close()

        # Save the data.
        save_filename = 'galids_{}.hdf5'.format( self.tag )
        self.save_filepath = os.path.join( self.out_dir, save_filename )
        f = h5py.File( self.save_filepath )
        for key in ptrack_gal_ids.keys():
            f.create_dataset( key, data=ptrack_gal_ids[key] )

        # Store the main mt halo id
        # (as identified by the larges value at the lowest redshift)
        if self.main_mt_halo_id is None:
            try:
                indice = self.halo_data.mtree_halos[0].index.max()
            except AttributeError:
                self.halo_data.get_mtree_halos(
                    self.mtree_halos_index, self.halo_file_tag )
                indice = self.halo_data.mtree_halos[0].index.max()
            m_vir_z0 = self.halo_data.get_mtree_halo_quantity(
                quantity='Mvir', indice=indice,
                index=self.mtree_halos_index, tag=self.halo_file_tag )
            f.attrs['main_mt_halo_id'] = m_vir_z0.argmax()
        else:
            f.attrs['main_mt_halo_id'] = self.main_mt_halo_id

        utilities.save_parameters( self, f )

        # Save the current code version
        f.attrs['linefinder_version'] = utilities.get_code_version( self )
        f.attrs['galaxy_dive_version'] = utilities.get_code_version(
            galaxy_linker, instance_type='module' )

        f.close()
