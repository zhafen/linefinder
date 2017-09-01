#self.!/usr/bin/env python
'''Tools for tracking particles.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import gc
import h5py
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time

import galaxy_diver.read_data.snapshot as read_snapshot
import galaxy_diver.utils.constants as constants
import galaxy_diver.utils.mp_utils as mp_utils
import galaxy_diver.utils.utilities as utilities

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class ParticleTracker( object ):
  '''Searches IDs across snapshots, then saves the results.'''

  @utilities.store_parameters
  def __init__( self,
    out_dir,
    tag,
    sdir = default,
    p_types = default,
    snum_start = default,
    snum_end = default,
    snum_step = default,
    n_processors = 1,
    ):
    '''Setup the ID Finder. Looks for data in the form of "out_dir/ids_tag.hdf5"

    Args:
      out_dir (str) :
        Output data directory. Also the directory for the file the ids to track should be in.

      tag (str) :
        Identifying tag. Currently must be put in manually. Should be the same for all stages of the pipeline.

      sdir (str, optional):
        Simulation data directory. Defaults to same directory the IDs are from.

      p_types (list of ints, optional):
        The particle data types to include. Defaults to the values used in the IDs.

      snum_start (int, optional):
        Starting snapshot. Defaults to the value used in the IDs.

      snum_end (int, optional):
        End snapshot. Defaults to same directory the IDs are from.

      snum_step (int, optional):
        How many snapshots to jump over? Defaults to same directory the IDs are from.

      n_processors (int, optional) :
        Number of processors to use.
    '''

    pass

  ########################################################################

  def save_particle_tracks( self ):
    '''Loop over all redshifts, get the data, and save the particle tracks.'''

    time_start = time.time()

    print "########################################################################"
    print "Starting Tracking!"
    print "########################################################################"
    print "Tracking particle data from this directory:\n    {}".format( self.sdir )
    print "Data will be saved here:\n    {}".format( self.out_dir )
    sys.stdout.flush()

    # Get the target ids
    self.get_target_ids()

    # Loop overall redshift snapshots and get the data out
    if self.n_processors > 1:
      self.ptrack = self.get_tracked_data_parallel()
    else:
      self.ptrack = self.get_tracked_data()

    # Write particle data to the file
    self.write_tracked_data()

    time_end = time.time()

    print "########################################################################"
    print "Done Tracking!"
    print "########################################################################"
    print "Output file saved as:\n    {}".format( self.outname )
    print "Took {:.3g} seconds, or {:.3g} seconds per particle!".format( time_end - time_start, (time_end - time_start) / self.ntrack )

  ########################################################################

  def get_target_ids( self ):
    '''Open the file containing the target ids and retrieve them.

    Modifies:
      self.target_ids (np.array): Fills the array.
      self.target_child_ids (np.array, optional): Fills the array, if child_ids are included.
    '''

    id_filename = 'ids_{}.hdf5'.format( self.tag )
    self.id_filepath = os.path.join( self.out_dir, id_filename )

    f = h5py.File( self.id_filepath, 'r' )

    # Load in the parameters
    def replace_default_attr( attr_name ):
      attr = getattr( self, attr_name )
      if attr is default:
        if attr_name == 'sdir':
          attr = f['parameters/snapshot_parameters'].attrs[attr_name]
        else:
          attr = f['parameters'].attrs[attr_name]
        setattr( self, attr_name, attr )
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
      ptrack (dict): Structure to hold particle tracks.
                     Structure is... ptrack ['varname'] [particle i, snap j, k component]
    '''

    self.snaps = np.arange( self.snum_end, self.snum_start-1, -self.snum_step )
    nsnap = self.snaps.size       # number of redshift snapshots that we follow back

    # Choose between single or double precision.
    myfloat = 'float32'

    self.ntrack = self.target_ids.size

    print "Tracking {} particles...".format( self.ntrack )
    sys.stdout.flush()

    ptrack = {
      'redshift':np.zeros( nsnap, dtype=myfloat ), 
      'snum':np.zeros( nsnap, dtype='int16' ),
      'ID':np.zeros( self.ntrack, dtype='int64' ), 
      'PType':np.zeros( self.ntrack, dtype=('int8',(nsnap,)) ),
      'Den':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ), 
      'SFR':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'T':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'Z':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'M':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'P':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,3)) ),
      'V':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,3)) ), 
    }

    ptrack['ID'] = self.target_ids
    if self.target_child_ids is not None:
      ptrack['ChildID'] = self.target_child_ids

    j = 0

    for snum in self.snaps:

      time_1 = time.time()

      id_finder = IDFinder()
      dfid, redshift, self.attrs = id_finder.find_ids( self.sdir, snum, self.p_types, self.target_ids, \
                                           target_child_ids=self.target_child_ids, )

      ptrack['redshift'][j] = redshift
      ptrack['snum'][j] = snum
      ptrack['PType'][:,j] = dfid['PType'].values
      ptrack['Den'][:,j] = dfid['Den'].values                                                           # cm^(-3)
      ptrack['SFR'][:,j] = dfid['SFR'].values                                                           # Msun / year   (stellar Age in Myr for star particles)
      ptrack['T'][:,j] = dfid['T'].values                                                               # Kelvin
      ptrack['Z'][:,j] = dfid['Z'].values                                                               # Zsun (metal mass fraction in Solar units)
      ptrack['M'][:,j] = dfid['M'].values                                                               # Msun (particle mass in solar masses)
      ptrack['P'][:,j,:] = np.array( [ dfid['P0'].values, dfid['P1'].values, dfid['P2'].values ] ).T    # kpc (physical)
      ptrack['V'][:,j,:] = np.array( [ dfid['V0'].values, dfid['V1'].values, dfid['V2'].values ] ).T    # km/s (peculiar - need to add H(a)*r contribution)

      j += 1

      gc.collect()          # helps stop leaking memory ?
      time_2 = time.time()

      # Print output information.
      print 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'.format(  snum, redshift, time_2 - time_1 )
      sys.stdout.flush()

    return ptrack

  ########################################################################

  def get_tracked_data_parallel( self ):
    '''Loop overall redshift snapshots, and get the data. This is the parallelized version.

    Returns:
      ptrack (dict): Structure to hold particle tracks.
                     Structure is... ptrack ['varname'] [particle i, snap j, k component]
    '''

    self.snaps = np.arange( self.snum_end, self.snum_start-1, -self.snum_step )
    nsnap = self.snaps.size       # number of redshift snapshots that we follow back

    # Choose between single or double precision.
    myfloat = 'float32'

    self.ntrack = self.target_ids.size
    print "Tracking {} particles...".format( self.ntrack )
    sys.stdout.flush()

    def get_tracked_data_snapshot( args ):

      i, snum = args

      time_1 = time.time()

      id_finder = IDFinder()
      dfid, redshift, attrs = id_finder.find_ids( self.sdir, snum, self.p_types, self.target_ids, \
                                           target_child_ids=self.target_child_ids, )

      # Maybe helps stop leaking memory
      del id_finder
      gc.collect()

      time_2 = time.time()

      # Print output information.
      print 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'.format( snum, redshift, time_2 - time_1 )
      sys.stdout.flush()

      return i, dfid, redshift, attrs, snum

    all_args = [ arg for arg in enumerate( self.snaps ) ]

    tracked_data_snapshots = mp_utils.parmap( get_tracked_data_snapshot, all_args, self.n_processors )

    ptrack = {
      'redshift' : np.zeros( nsnap, dtype=myfloat ), 
      'snum' : np.zeros( nsnap, dtype='int16' ),
      'ID' : np.zeros( self.ntrack, dtype='int64' ), 
      'PType' : np.zeros( self.ntrack, dtype=('int8',(nsnap,)) ),
      'Den' : np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ), 
      'SFR' : np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'T' : np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'Z' : np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'M' : np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'P' : np.zeros( self.ntrack, dtype=(myfloat,(nsnap,3)) ),
      'V' : np.zeros( self.ntrack, dtype=(myfloat,(nsnap,3)) ), 
    }

    ptrack['ID'] = self.target_ids
    if self.target_child_ids is not None:
      ptrack['ChildID'] = self.target_child_ids

    assert len( tracked_data_snapshots ) == self.snaps.size,\
      "Unequal sizes, snapshot likely skipped, likely due to a MemoryError!"

    for tracked_data_snapshot in tracked_data_snapshots:

      j, dfid, redshift, self.attrs, snum = tracked_data_snapshot

      ptrack['redshift'][j] = redshift
      ptrack['snum'][j] = snum
      ptrack['PType'][:,j] = dfid['PType'].values
      ptrack['Den'][:,j] = dfid['Den'].values                                                           # cm^(-3)
      ptrack['SFR'][:,j] = dfid['SFR'].values                                                           # Msun / year   (stellar Age in Myr for star particles)
      ptrack['T'][:,j] = dfid['T'].values                                                               # Kelvin
      ptrack['Z'][:,j] = dfid['Z'].values                                                               # Zsun (metal mass fraction in Solar units)
      ptrack['M'][:,j] = dfid['M'].values                                                               # Msun (particle mass in solar masses)
      ptrack['P'][:,j,:] = np.array( [ dfid['P0'].values, dfid['P1'].values, dfid['P2'].values ] ).T    # kpc (physical)
      ptrack['V'][:,j,:] = np.array( [ dfid['V0'].values, dfid['V1'].values, dfid['V2'].values ] ).T    # km/s (peculiar - need to add H(a)*r contribution)

    return ptrack

  ########################################################################

  def write_tracked_data( self ):
    '''Write tracks to a file.'''

    # Make sure the output location exists
    if not os.path.exists( self.out_dir ):
      os.mkdir( self.out_dir )

    self.outname = 'ptracks_{}.hdf5'.format( self.tag )

    outpath =  os.path.join( self.out_dir, self.outname )

    f = h5py.File( outpath, 'w' )

    # Save data
    for keyname in self.ptrack.keys():
        f.create_dataset( keyname, data=self.ptrack[keyname] )

    # Save the attributes
    for key in self.attrs.keys():
      f.attrs[key] = self.attrs[key]

    # Save the data parameters too, as part of a group
    grp = f.create_group('parameters')
    for parameter in self.stored_parameters:
      grp.attrs[parameter] = getattr( self, parameter )

    # Save the number of processors
    grp.attrs['n_processors'] = self.n_processors

    # Save the current code version
    f.attrs['pathfinder_version'] = utilities.get_code_version( self )
    f.attrs['galaxy_diver_version'] = utilities.get_code_version( read_snapshot, instance_type='module' )

    f.close()

########################################################################
########################################################################

class IDFinder( object ):
  '''Finds target ids in a single snapshot.'''

  def __init__( self ):
    pass

  ########################################################################

  def find_ids( self, sdir, snum, p_types, target_ids, target_child_ids=None, ):
    '''Find the information for particular IDs in a given snapshot, ordered by the ID list you pass.

    Args:
      sdir (str): The targeted simulation directory.
      snum (int): The snapshot to find the IDs for.
      p_types (list of ints): Which particle types to target.
      target_ids (np.array): The particle IDs you want to find.

    Returns:
      dfid (pandas.DataFrame): Dataframe for the selected IDs, ordered by target_ids.
          Contains standard particle information, e.g. position, metallicity, etc.
      redshift (float): Redshift of the snapshot.
      attrs (dict): Dictionary of attributes of the snapshot
    '''

    # Store the target ids for easy access.
    self.sdir = sdir
    self.snum = snum
    self.p_types = p_types
    self.target_ids = target_ids
    if target_child_ids is not None:
      self.target_child_ids = target_child_ids

    # Make a big list of the relevant particle data, across all the particle data types
    self.concatenate_particle_data()

    # Find target_ids 
    self.dfid = self.select_ids()

    return self.dfid, self.redshift, self.attrs

  ########################################################################

  def concatenate_particle_data(self, verbose=False):
    '''Get all the particle data for the snapshot in one big array, to allow searching through it.

    Args:
      verbose (bool): If True, print additional information.

    Modifies:
      self.full_snap_data (dict): A dictionary of the concatenated particle data.
    '''

    if verbose:
      print 'Reading data...'
      sys.stdout.flush()

    full_snap_data = {
      'ID' : [],
      'PType' : [],
      'Den' : [],
      'SFR' : [],
      'T' : [],
      'Z' : [],
      'M' : [],
      'P0' : [],
      'P1' : [],
      'P2' : [],
      'V0' : [],
      'V1' : [],
      'V2' : [],
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

      P = read_snapshot.readsnap( self.sdir, self.snum, p_type, load_additional_ids=load_additional_ids, cosmological=1, skip_bh=1, header_only=0 )

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
        print '       ...  ', pnum, '   type', p_type , ' particles'

      if 'rho' in P:
          Den = P['rho']*constants.UNITDENSITY_IN_NUMDEN
      else:
          Den = [0.,]*pnum

      if 'sfr' in P:
          sfr = P['sfr']
      else:
          sfr = [0.,]*pnum

      if 'u' in P:
          T = read_snapshot.gas_temperature( P['u'], P['ne'] )
      else:
          T = [0.,]*pnum

      thistype = np.zeros(pnum,dtype='int8')
      thistype.fill(p_type)

      full_snap_data['ID'].append( P['id'] )
      full_snap_data['PType'].append( thistype )
      full_snap_data['Den'].append( Den )
      full_snap_data['SFR'].append( sfr )
      full_snap_data['T'].append( T )
      full_snap_data['Z'].append( P['z'][:,0]/constants.Z_MASSFRAC_SUN )
      full_snap_data['M'].append( P['m']*constants.UNITMASS_IN_MSUN )
      full_snap_data['P0'].append( P['p'][:,0] )
      full_snap_data['P1'].append( P['p'][:,1] )
      full_snap_data['P2'].append( P['p'][:,2] )
      full_snap_data['V0'].append( P['v'][:,0] )
      full_snap_data['V1'].append( P['v'][:,1] )
      full_snap_data['V2'].append( P['v'][:,2] )

      if hasattr( self, 'target_child_ids' ):
        full_snap_data['ChildID'].append( P['child_id'] )

    time_end = time.time()

    if verbose:
      print 'readsnap done in ... {:.3g} seconds'.format( time_end - time_start )

    # Convert to numpy arrays
    for key in full_snap_data.keys():
      full_snap_data[key] = np.concatenate( full_snap_data[key] )

    self.full_snap_data = full_snap_data

  ########################################################################

  def select_ids(self):
    '''Function for selecting the targeted ids from the snapshot.'''

    # Setup the index and the way to select the targeted ids
    if hasattr( self, 'target_child_ids' ):
      index = [ self.full_snap_data['ID'], self.full_snap_data['ChildID'] ]
      target_selection = list( zip( *[ self.target_ids, self.target_child_ids ] ) )
    else:
      index = self.full_snap_data['ID']
      target_selection = self.target_ids

    # Make a data frame.
    df = pd.DataFrame( data=self.full_snap_data, index=index )      

    # Sort for faster indexing
    df.sort()

    # Make a data frame selecting only the target ids
    dfid = df.loc[ target_selection ]

    # When particle IDs can't be found the values are automatically replaced by NaNs. This is very good, but
    # we want to preserve the ids and child ids for the particles we couldn't find.
    if hasattr( self, 'target_child_ids' ):
      dfid['ID'] = dfid.index.get_level_values( 0 )
      dfid['ChildID'] = dfid.index.get_level_values( 1 )

    assert len( dfid['ID'] ) == len( self.target_ids ), \
      "Snapshot {} failed, len( df ) = {}, len( self.target_ids ) = {}".format( self.snum, len( dfid['ID'] ),
                                                                                len( self.target_ids ) )

    return dfid
