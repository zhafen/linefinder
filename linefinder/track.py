#!/usr/bin/env python
'''Tools for tracking particles.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import gc
import h5py
import inspect
import jug
import numpy as np
import os
import pandas as pd
import sys
import time

import galaxy_dive.read_data.snapshot as read_snapshot
import galaxy_dive.utils.constants as constants
import galaxy_dive.utils.mp_utils as mp_utils
import galaxy_dive.utils.utilities as utilities

from . import config as linefinder_config

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################


class ParticleTracker( object ):
    '''Searches IDs across snapshots, then saves the results.
    '''

    @utilities.store_parameters
    def __init__(
        self,
        out_dir,
        tag,
        ids_tag = default,
        sdir = default,
        p_types = default,
        snum_start = default,
        snum_end = default,
        snum_step = default,
        n_processors = 1,
        custom_fns = None,
    ):
        '''Setup the ID Finder. Looks for data in the form of "out_dir/ids_tag.hdf5"

        Args:
            out_dir (str) :
                Output data directory. Also the directory for the file the ids
                to track should be in.

            tag (str) :
                Identifying tag. Currently must be put in manually.

            ids_tag (str) :
                Identifying tag for the ids. Defaults to tag.

            sdir (str, optional):
                Simulation data directory. Defaults to same directory the IDs
                are from.

            p_types (list of ints, optional):
                The particle data types to include. Defaults to the values used
                in the IDs.

            snum_start (int, optional):
                Starting snapshot. Defaults to the value used in the IDs.

            snum_end (int, optional):
                End snapshot. Defaults to same directory the IDs are from.

            snum_step (int, optional):
                How many snapshots to jump over? Defaults to same directory the
                IDs are from.

            n_processors (int, optional) :
                Number of processors to use.

            custom_fns (list of functions):
                If not None, a list of functions to calculate additional
                derived products for the selected IDs. Arguments should be a
                pandas DataFrame named dfid that contains the data of the
                selected IDs, and a pandas DataFrame named df that contains
                the full snapshot data. The derived products should be stored
                as new columns in dfid.
        '''

        if self.ids_tag is default:
            self.ids_tag = tag

        pass

    ########################################################################

    def save_particle_tracks( self ):
        '''Loop over all redshifts, get the data, and save the particle tracks.
        '''

        time_start = time.time()

        print( "#" * 80 )
        print( "Starting Tracking!" )
        print( "#" * 80 )
        print( "Tracking particle data from this directory:\n    {}".format(
                self.sdir
            )
        )
        print( "Data will be saved here:\n    {}".format( self.out_dir ) )
        sys.stdout.flush()

        # Get the target ids
        self.get_target_ids()

        # Loop overall redshift snapshots and get the data out
        if self.n_processors > 1:
            formatted_data = self.get_tracked_data_parallel()
        else:
            formatted_data = self.get_tracked_data()

        # Write particle data to the file
        self.write_tracked_data( formatted_data )

        time_end = time.time()

        print( "#" * 80 )
        print( "Done Tracking!" )
        print( "#" * 80 )
        print( "Output file saved as:\n    {}".format( self.outname ) )
        print( "Took {:.3g} seconds, or {:.3g} seconds per particle!".format(
                time_end - time_start, (time_end - time_start) / self.ntrack
            )
        )

    ########################################################################

    def save_particle_tracks_jug( self ):
        '''Loop over all redshifts, get the data, and save the particle tracks.
        '''

        print( "#" * 80 )
        print( "Starting Tracking!" )
        print( "#" * 80 )

        # Get the target ids
        self.get_target_ids()

        tracked_data_snapshots = self.get_tracked_data_jug()

        formatted_data = jug.Task(
            self.format_tracked_data, tracked_data_snapshots )

        # Write particle data to the file
        jug.Task( self.write_tracked_data, formatted_data )

        jug.barrier()

    ########################################################################

    def get_target_ids( self ):
        '''Open the file containing the target ids and retrieve them.

        Modifies:
            self.target_ids (np.array):
                Fills the array.

            self.target_child_ids (np.array, optional): Fills the array, if
                child_ids are included.
        '''

        id_filename = 'ids_{}.hdf5'.format( self.ids_tag )
        self.id_filepath = os.path.join( self.out_dir, id_filename )

        f = h5py.File( self.id_filepath, 'r' )

        # Load in the parameters
        def replace_default_attr( attr_name ):
            attr = getattr( self, attr_name )
            if attr is default:
                try:
                    if attr_name == 'sdir':
                        attr = utilities.check_and_decode_bytes(
                            f['parameters/snapshot_parameters'].attrs[attr_name]
                        )
                    else:
                        attr = utilities.check_and_decode_bytes(
                            f['parameters'].attrs[attr_name]
                        )
                    setattr( self, attr_name, attr )
                except KeyError:
                    raise LookupError(
                        'Cannot fallback to default parameters because ' + \
                        '{} does not include default parameters.'.format(
                            id_filename,
                        )
                    )
        replace_default_attr( 'sdir' )
        replace_default_attr( 'p_types' )
        replace_default_attr( 'snum_start' )
        replace_default_attr( 'snum_end' )
        replace_default_attr( 'snum_step' )

        # Load in the data
        for key in f.keys():
            if key != 'parameters':
                setattr( self, key, f[key][...] )

        # If there aren't target child IDs, make note of that
        if 'target_child_ids' not in f.keys():
            self.target_child_ids = None

    ########################################################################

    def get_tracked_data( self ):
        '''Loop overall redshift snapshots, and get the data.

        Returns:
            ptrack (dict):
                Structure to hold particle tracks.
                Structure is...
                ptrack ['varname'] [particle i, snap j, k component]
        '''

        self.snaps = np.arange( self.snum_end, self.snum_start - 1,
                                -self.snum_step )
        # number of redshift snapshots that we follow back
        nsnap = self.snaps.size

        # Choose between single or double precision.
        myfloat = 'float32'

        self.ntrack = self.target_ids.size

        print( "Tracking {} particles...".format( self.ntrack ) )
        sys.stdout.flush()

        ptrack = {
            'redshift': np.zeros( nsnap, dtype=myfloat ),
            'snum': np.zeros( nsnap, dtype='int16' ),
            'ID': np.zeros( self.ntrack, dtype='int64' ),
            'PType': np.zeros( self.ntrack, dtype=('int8', (nsnap,)) ),
            'Den': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'SFR': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'T': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'Z': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'M': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'P': np.zeros( self.ntrack, dtype=(myfloat, (nsnap, 3)) ),
            'V': np.zeros( self.ntrack, dtype=(myfloat, (nsnap, 3)) ),
        }

        ptrack['ID'] = self.target_ids
        if self.target_child_ids is not None:
            ptrack['ChildID'] = self.target_child_ids

        j = 0
        for snum in self.snaps:

            time_1 = time.time()

            id_finder = IDFinder()
            dfid, redshift, attrs = id_finder.find_ids(
                self.sdir,
                snum,
                self.p_types,
                self.target_ids,
                target_child_ids = self.target_child_ids,
                custom_fns = self.custom_fns,
            )

            ptrack['redshift'][j] = redshift
            ptrack['snum'][j] = snum
            ptrack['PType'][:, j] = dfid['PType'].values
            # cm^(-3)
            ptrack['Den'][:, j] = dfid['Den'].values
            # Msun / year   (stellar Age in Myr for star particles)
            ptrack['SFR'][:, j] = dfid['SFR'].values
            # Kelvin
            ptrack['T'][:, j] = dfid['T'].values
            # Zsun (metal mass fraction in Solar units)
            ptrack['Z'][:, j] = dfid['Z'].values
            # Msun (particle mass in solar masses)
            ptrack['M'][:, j] = dfid['M'].values
            # kpc (physical)
            ptrack['P'][:, j, :] = np.array(
                [ dfid['P0'].values, dfid['P1'].values, dfid['P2'].values ]
            ).T
            # km/s (peculiar - need to add H(a)*r contribution)
            ptrack['V'][:, j, :] = np.array(
                [ dfid['V0'].values, dfid['V1'].values, dfid['V2'].values ]
            ).T

            # Include custom derived products
            for data_key in dfid.columns:

                # Skip existing data
                if data_key in ptrack.keys():
                    continue

                # Store the data
                try:
                    ptrack[data_key][:, j] = dfid[data_key].values
                except KeyError:
                    ptrack[data_key] = np.zeros(
                        self.ntrack,
                        dtype=( myfloat, (nsnap,) ),
                    )
                    ptrack[data_key][:, j] = dfid[data_key].values


            if 'Potential' in dfid.keys():
                try:
                    ptrack['Potential'][:, j] = dfid['Potential'].values
                except KeyError:
                    ptrack['Potential'] = np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) )
                    ptrack['Potential'][:, j] = dfid['Potential'].values

            j += 1

            gc.collect()          # helps stop leaking memory ?
            time_2 = time.time()

            # Print output information.
            print( 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'\
                .format(  snum, redshift, time_2 - time_1 )
            )
            sys.stdout.flush()

        return ptrack, attrs

    ########################################################################

    def get_tracked_data_parallel( self ):
        '''Loop overall redshift snapshots, and get the data. This is the
        parallelized version.

        Returns:
            ptrack (dict):
                Structure to hold particle tracks.
                Structure is...
                ptrack ['varname'] [particle i, snap j, k component]
        '''

        self.snaps = np.arange(
            self.snum_end, self.snum_start - 1, -self.snum_step )
        # number of redshift snapshots that we follow back
        nsnap = self.snaps.size

        # Choose between single or double precision.
        myfloat = 'float32'

        self.ntrack  =  self.target_ids.size
        print( "Tracking {} particles...".format( self.ntrack ) )
        sys.stdout.flush()

        def get_tracked_data_snapshot( args ):

            i, snum = args

            time_1 = time.time()

            id_finder = IDFinder()
            dfid, redshift, attrs = id_finder.find_ids(
                self.sdir,
                snum,
                self.p_types,
                self.target_ids,
                target_child_ids = self.target_child_ids,
                custom_fns = self.custom_fns,
            )

            # Maybe helps stop leaking memory
            del id_finder
            gc.collect()

            time_2 = time.time()

            # Print output information.
            print( 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'\
                .format( snum, redshift, time_2 - time_1 )
            )
            sys.stdout.flush()

            return i, dfid, redshift, attrs, snum

        all_args = [ arg for arg in enumerate( self.snaps ) ]

        tracked_data_snapshots = mp_utils.parmap(
            get_tracked_data_snapshot, all_args, self.n_processors )

        ptrack = {
            'redshift': np.zeros( nsnap, dtype=myfloat ),
            'snum': np.zeros( nsnap, dtype='int16' ),
            'ID': np.zeros( self.ntrack, dtype='int64' ),
            'PType': np.zeros( self.ntrack, dtype=('int8', (nsnap,)) ),
            'Den': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'SFR': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'T': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'Z': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'M': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'P': np.zeros( self.ntrack, dtype=(myfloat, (nsnap, 3)) ),
            'V': np.zeros( self.ntrack, dtype=(myfloat, (nsnap, 3)) ),
        }

        ptrack['ID'] = self.target_ids
        if self.target_child_ids is not None:
            ptrack['ChildID'] = self.target_child_ids

        assert len( tracked_data_snapshots ) == self.snaps.size,\
            "Unequal sizes, snapshot likely skipped," + \
            " likely due to a MemoryError!"

        for tracked_data_snapshot in tracked_data_snapshots:

            j, dfid, redshift, attrs, snum = tracked_data_snapshot

            ptrack['redshift'][j] = redshift
            ptrack['snum'][j] = snum
            ptrack['PType'][:, j] = dfid['PType'].values
            # cm^(-3)
            ptrack['Den'][:, j] = dfid['Den'].values
            # Msun / year   (stellar Age in Myr for star particles)
            ptrack['SFR'][:, j] = dfid['SFR'].values
            # Kelvin
            ptrack['T'][:, j] = dfid['T'].values
            # Zsun (metal mass fraction in Solar units)
            ptrack['Z'][:, j] = dfid['Z'].values
            # Msun (particle mass in solar masses)
            ptrack['M'][:, j] = dfid['M'].values
            # kpc (physical)
            ptrack['P'][:, j, :] = np.array(
                [ dfid['P0'].values, dfid['P1'].values, dfid['P2'].values ]
            ).T
            # km/s (peculiar - need to add H(a)*r contribution)
            ptrack['V'][:, j, :] = np.array(
                [ dfid['V0'].values, dfid['V1'].values, dfid['V2'].values ]
            ).T

            # Include custom derived products
            for data_key in dfid.columns:

                # Skip existing data
                if data_key in ptrack.keys():
                    continue

                # Store the data
                try:
                    ptrack[data_key][:, j] = dfid[data_key].values
                except KeyError:
                    ptrack[data_key] = np.zeros(
                        self.ntrack,
                        dtype=( myfloat, (nsnap,) ),
                    )
                    ptrack[data_key][:, j] = dfid[data_key].values


        return ptrack, attrs

    ########################################################################

    def get_tracked_data_jug( self ):
        '''Loop overall redshift snapshots, and get the data. This is the
        parallelized version that uses Jug

        Returns:
            ptrack (dict):
                Structure to hold particle tracks.
                Structure is...
                ptrack ['varname'] [particle i, snap j, k component]
        '''

        self.snaps = np.arange(
            self.snum_end, self.snum_start - 1, -self.snum_step )

        self.ntrack = self.target_ids.size
        print( "Tracking {} particles...".format( self.ntrack ) )
        sys.stdout.flush()

        def get_tracked_data_snapshot( args ):

            i, snum = args

            time_1 = time.time()

            id_finder = IDFinder()
            dfid, redshift, attrs = id_finder.find_ids(
                self.sdir,
                snum,
                self.p_types,
                self.target_ids,
                target_child_ids = self.target_child_ids,
                custom_fns = self.custom_fns,
            )

            # Maybe helps stop leaking memory
            del id_finder
            gc.collect()

            time_2 = time.time()

            # Print output information.
            print(
                 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'\
                .format( snum, redshift, time_2 - time_1 )
            )
            sys.stdout.flush()

            return i, dfid, redshift, attrs, snum

        tracked_data_snapshots = []
        for args in enumerate( self.snaps ):

            tracked_data = jug.Task( get_tracked_data_snapshot, args )

            tracked_data_snapshots.append( tracked_data )

        return tracked_data_snapshots

    ########################################################################

    def format_tracked_data( self, tracked_data_snapshots ):
        '''Format data for storage.

        Args:
            tracked_data_snapshots (list) :
                List of unsorted data.

        Returns:
            ptrack (dict) :
                Formatted data.

            attrs (dict) :
                Formatted data attributes.
        '''

        # number of redshift snapshots that we follow back
        nsnap = self.snaps.size

        # Choose between single or double precision.
        myfloat = 'float32'

        ptrack = {
            'redshift': np.zeros( nsnap, dtype=myfloat ),
            'snum': np.zeros( nsnap, dtype='int16' ),
            'ID': np.zeros( self.ntrack, dtype='int64' ),
            'PType': np.zeros( self.ntrack, dtype=('int8', (nsnap,)) ),
            'Den': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'SFR': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'T': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'Z': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'M': np.zeros( self.ntrack, dtype=(myfloat, (nsnap,)) ),
            'P': np.zeros( self.ntrack, dtype=(myfloat, (nsnap, 3)) ),
            'V': np.zeros( self.ntrack, dtype=(myfloat, (nsnap, 3)) ),
        }

        ptrack['ID'] = self.target_ids
        if self.target_child_ids is not None:
            ptrack['ChildID'] = self.target_child_ids

        for tracked_data_snapshot in tracked_data_snapshots:

            j, dfid, redshift, attrs, snum = tracked_data_snapshot

            ptrack['redshift'][j] = redshift
            ptrack['snum'][j] = snum
            ptrack['PType'][:, j] = dfid['PType'].values
            # cm^(-3)
            ptrack['Den'][:, j] = dfid['Den'].values
            # Msun / year   (stellar Age in Myr for star particles)
            ptrack['SFR'][:, j] = dfid['SFR'].values
            # Kelvin
            ptrack['T'][:, j] = dfid['T'].values
            # Zsun (metal mass fraction in Solar units)
            ptrack['Z'][:, j] = dfid['Z'].values
            # Msun (particle mass in solar masses)
            ptrack['M'][:, j] = dfid['M'].values
            # kpc (physical)
            ptrack['P'][:, j, :] = np.array(
                [ dfid['P0'].values, dfid['P1'].values, dfid['P2'].values ]
            ).T
            # km/s (peculiar - need to add H(a)*r contribution)
            ptrack['V'][:, j, :] = np.array(
                [ dfid['V0'].values, dfid['V1'].values, dfid['V2'].values ]
            ).T

            # Include custom derived products
            for data_key in dfid.columns:

                # Skip existing data
                if data_key in ptrack.keys():
                    continue

                # Store the data
                try:
                    ptrack[data_key][:, j] = dfid[data_key].values
                except KeyError:
                    ptrack[data_key] = np.zeros(
                        self.ntrack,
                        dtype=( myfloat, (nsnap,) ),
                    )
                    ptrack[data_key][:, j] = dfid[data_key].values

        return ptrack, attrs

    ########################################################################
    ########################################################################

    def write_tracked_data( self, formatted_data ):
        '''Write tracks to a file.

        Args:
            formatted_data (dict) :
                Formatted particle track data.

            attrs (dict):
                Particle track data attributes.
        '''

        ptrack, attrs = formatted_data

        # Make sure the output location exists
        if not os.path.exists( self.out_dir ):
            os.mkdir( self.out_dir )

        self.outname = 'ptracks_{}.hdf5'.format( self.tag )

        outpath = os.path.join( self.out_dir, self.outname )

        f = h5py.File( outpath, 'w' )

        # Save data
        for keyname in ptrack.keys():
                f.create_dataset( keyname, data=ptrack[keyname] )

        # Save the attributes
        for key in attrs.keys():
            f.attrs[key] = attrs[key]

        # Save the code of the custom_fns as a string too.
        if self.custom_fns is not None:
            self.custom_fns_str = [
                inspect.getsource( _ ) for _ in self.custom_fns
            ]

        utilities.save_parameters( self, f )

        # Save the current code version
        f.attrs['linefinder_version'] = utilities.get_code_version( self )
        f.attrs['galaxy_dive_version'] = utilities.get_code_version(
            read_snapshot, instance_type='module' )

        f.close()

########################################################################
########################################################################


class IDFinder( object ):
    '''Finds target ids in a single snapshot.
    '''

    def __init__( self ):
        pass

    ########################################################################

    def find_ids(
        self,
        sdir,
        snum,
        p_types,
        target_ids,
        target_child_ids = None,
        custom_fns = None,
    ):
        '''Find the information for particular IDs in a given snapshot, ordered
        by the ID list you pass.

        Args:
            sdir (str):
                The targeted simulation directory.

            snum (int):
                The snapshot to find the IDs for.

            p_types (list of ints):
                Which particle types to target.

            target_ids (np.array):
                The particle IDs you want to find.

            target_child_ids (np.array):
                The particle child IDs you want to find.

            custom_fns (list of functions):
                If not None, additional derived products you want to calculate
                for the selected IDs. Arguments should be a pandas DataFrame
                named dfid that contains the data of the selected IDs, and a
                pandas DataFrame named df that contains the full snapshot data.

        Returns:
            dfid (pandas.DataFrame):
                Dataframe for the selected IDs, ordered by
                target_ids. Contains standard particle information, e.g.
                position, metallicity, etc.

            redshift (float):
                Redshift of the snapshot.

            attrs (dict):
                Dictionary of attributes of the snapshot
        '''

        # Store the target ids for easy access.
        self.sdir = sdir
        self.snum = snum
        self.p_types = p_types
        self.target_ids = target_ids
        if target_child_ids is not None:
            self.target_child_ids = target_child_ids

        # Make a big list of the relevant particle data, across all the particle
        # data types
        self.concatenate_particle_data()

        # Find target_ids
        self.dfid, self.df = self.select_ids()

        # Apply additional functions
        if custom_fns is not None:
            self.dfid = self.apply_functions(
                custom_fns,
                self.dfid,
                self.df,
            )

        return self.dfid, self.redshift, self.attrs

    ########################################################################

    def concatenate_particle_data( self, verbose=False ):
        '''Get all the particle data for the snapshot in one big array, to allow
        searching through it.

        Args:
            verbose (bool): If True, print additional information.

        Modifies:
            self.full_snap_data (dict):
            A dictionary of the concatenated particle data.
        '''

        if verbose:
            print( 'Reading data...' )
            sys.stdout.flush()

        full_snap_data = {
            'ID': [],
            'PType': [],
            'Den': [],
            'SFR': [],
            'T': [],
            'Z': [],
            'M': [],
            'P0': [],
            'P1': [],
            'P2': [],
            'V0': [],
            'V1': [],
            'V2': [],
        }

        time_start = time.time()

        if hasattr( self, 'target_child_ids' ):
            load_additional_ids = True
            full_snap_data['ChildID'] = []
        else:
            load_additional_ids = False

        # Flag for saving header info
        saved_header_info = False

        for i, p_type in enumerate( self.p_types ):

            P = read_snapshot.readsnap(
                self.sdir,
                self.snum,
                p_type,
                load_additional_ids=load_additional_ids,
                cosmological=1,
                skip_bh=1,
                header_only=0
            )

            if P['k'] < 0:
                continue
            pnum = P['id'].size

            # On the first time through, save global information
            if not saved_header_info:

                # Save the redshift
                self.redshift = P['redshift']

                # Store the attributes to be used in the final data file.
                attrs_keys = [ 'omega_matter', 'omega_lambda', 'hubble' ]
                self.attrs = {}
                for key in attrs_keys:
                    self.attrs[key] = P[key]

                saved_header_info = True

            if verbose:
                print( '       ...  ', pnum, '   type', p_type, ' particles' )

            if 'rho' in P:
                    Den = P['rho'] * constants.UNITDENSITY_IN_NUMDEN
            else:
                    Den = [ linefinder_config.FLOAT_FILL_VALUE, ] * pnum

            if 'sfr' in P:
                    sfr = P['sfr']
            else:
                    sfr = [ linefinder_config.FLOAT_FILL_VALUE, ] * pnum

            if 'u' in P:
                    T = read_snapshot.gas_temperature( P['u'], P['ne'] )
            else:
                    T = [ linefinder_config.FLOAT_FILL_VALUE, ] * pnum

            thistype = np.zeros( pnum, dtype='int8' )
            thistype.fill(p_type)

            full_snap_data['ID'].append( P['id'] )
            full_snap_data['PType'].append( thistype )
            full_snap_data['Den'].append( Den )
            full_snap_data['SFR'].append( sfr )
            full_snap_data['T'].append( T )
            full_snap_data['Z'].append(
                P['z'][:, 0] / constants.Z_MASSFRAC_SUN )
            full_snap_data['M'].append( P['m'] * constants.UNITMASS_IN_MSUN )
            full_snap_data['P0'].append( P['p'][:, 0] )
            full_snap_data['P1'].append( P['p'][:, 1] )
            full_snap_data['P2'].append( P['p'][:, 2] )
            full_snap_data['V0'].append( P['v'][:, 0] )
            full_snap_data['V1'].append( P['v'][:, 1] )
            full_snap_data['V2'].append( P['v'][:, 2] )

            # Save the potential, if it exists
            if 'potential' in P.keys():
                try:
                    full_snap_data['Potential'].append( P['potential'] )
                except KeyError:
                    full_snap_data['Potential'] = [ P['potential'], ]

            if hasattr( self, 'target_child_ids' ):
                full_snap_data['ChildID'].append( P['child_id'] )

        time_end = time.time()

        if verbose:
            print( 'readsnap done in ... {:.3g} seconds'.format(
                    time_end - time_start
                )
            )

        # Convert to numpy arrays
        for key in full_snap_data.keys():
            full_snap_data[key] = np.concatenate( full_snap_data[key] )

        self.full_snap_data = full_snap_data

    ########################################################################

    def select_ids(self):
        '''Function for selecting the targeted ids from the snapshot.'''

        # Setup the index and the way to select the targeted ids
        if hasattr( self, 'target_child_ids' ):
            index = [ self.full_snap_data['ID'],
                      self.full_snap_data['ChildID'] ]
            target_selection = list(
                zip( *[ self.target_ids, self.target_child_ids ] ) )
        else:
            index = self.full_snap_data['ID']
            target_selection = self.target_ids

        # Make a data frame.
        df = pd.DataFrame( data=self.full_snap_data, index=index )

        # Sort for faster indexing
        df = df.sort_index()

        # Make a data frame selecting only the target ids
        dfid = df.loc[ target_selection ]

        # When particle IDs can't be found the values are automatically replaced
        # by NaNs. This is very good, but
        # we want to preserve the ids and child ids for the particles we
        # couldn't find.
        if hasattr( self, 'target_child_ids' ):
            dfid['ID'] = dfid.index.get_level_values( 0 )
            dfid['ChildID'] = dfid.index.get_level_values( 1 )

        assert len( dfid['ID'] ) == len( self.target_ids ), \
            "Snapshot {} failed, len( df ) = {}, len( self.target_ids ) = {}".\
            format( self.snum, len( dfid['ID'] ), len( self.target_ids ) )

        return dfid, df

    ########################################################################

    def apply_functions( self, fns, dfid, df ):
        '''Calculate and store the results of arbitrary functions of the full
        snapshot data and the subset of the snapshot data containing the IDs.
        
        Args:
            fns (list of fns):
                Functions to apply to the data. Should accept `dfid` and `df`
                as arguments, and modify dfid as desired. Can optionally accept
                an argument `id_finder`, which is the active instance of
                IDFinder, allowing for use in the functions of e.g. self.snum
                and self.attrs.

            dfid (pandas DataFrame):
                Contains attributes of the selected ID particles. Indexed by
                the ID.

            df (pandas DataFrame):
                Contains attributes of the full snapshot data. Indexed by the
                ID.

        Returns:
            pandas DataFrame:
                Modified dfid from having the fns applied to it. As a
                precaution, modifying any of the original columns will raise
                an error.
        '''

        # Make a copy before modifying
        original_dfid = copy.deepcopy( dfid )

        # Apply the functions
        for fn in fns:

            arg_names = fn.__code__.co_varnames

            assert (
                ( 'dfid' in arg_names ) and
                ( 'df' in arg_names )
            ), "Custom functions must use dfid and df as arguments."

            if 'id_finder' in arg_names:
                dfid = fn( dfid=dfid, df=df, id_finder=self )
            else:
                dfid = fn( dfid=dfid, df=df )

        # Check that the original data is intact
        for key in original_dfid.columns:
            np.testing.assert_allclose(
                original_dfid[key],
                dfid[key],
                err_msg = "A custom function modified the {} column of the data, which is part of the raw data!".format( key ),
            )

        return dfid
