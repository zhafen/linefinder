#self.!/usr/bin/env python
'''Tools for tracking particles.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import gc
import h5py
import numpy as np
import os
import pandas as pd
import sys
import time

import readsnap

########################################################################
########################################################################

class ParticleTracker( object ):
  '''Searches IDs across snapshots, then saves the results.'''

  def __init__( self, host_halo=False, **kwargs ):
    '''Setup the ID Finder. Looks for data in the form of "out_dir/ids_tag.hdf5"

    Args:
      host_halo (bool): Whether or not to include halo data for tracked particles during the tracking process.

    Keyword Args:
      Input Data Parameters:
        sdir (str): Simulation data directory.
        types (list of ints): The particle data types to include.
        snap_ini (int): Starting snapshot
        snap_end (int): End snapshot
        snap_step (int): How many snapshots to jump over?

      Analysis Parameters:
        tag (str): Identifying tag. Currently must be put in manually. Should be the same for all stages of the pipeline.
        outdir (str): Output data directory. Also the directory for the file the ids to track should be in.
    '''

    # Save the kwargs
    self.kwargs = kwargs

    # Save the arguments that have default values
    self.host_halo = host_halo

  ########################################################################

  def save_particle_tracks( self ):
    '''Loop over all redshifts, get the data, and save the particle tracks.'''

    time_start = time.time()

    print "########################################################################"
    print "Starting Tracking!"
    print "########################################################################"
    print "Tracking particle data from this directory:\n    {}".format( self.kwargs['sdir'] )
    print "Data will be saved here:\n    {}".format( self.kwargs['outdir'] )

    # Get the target ids
    self.get_target_ids()

    # Loop overall redshift snapshots and get the data out
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

    id_filename = 'ids_{}.hdf5'.format( self.kwargs['tag'] )
    id_filepath = os.path.join( self.kwargs['outdir'], id_filename )

    f = h5py.File( id_filepath, 'r' )

    # Load in the data
    for key in f.keys():
      setattr( self, key, f[key][...] )

    # If there aren't target child IDs, make note of that
    if 'target_child_ids' not in f.keys():
      self.target_child_ids = None

    # Make sure our simulation directory matches up
    assert os.path.samefile( self.kwargs['sdir'], f.attrs['sdir'] )

  ########################################################################

  def get_tracked_data( self ):
    '''Loop overall redshift snapshots, and get the data.

    Returns:
      ptrack (dict): Structure to hold particle tracks.
                     Structure is... ptrack ['varname'] [particle i, snap j, k component]
    '''
    self.snaps = np.arange( self.kwargs['snap_end'], self.kwargs['snap_ini']-1, -self.kwargs['snap_step'] )
    nsnap = self.snaps.size       # number of redshift snapshots that we follow back

    # Legacy of something Daniel encountered. Don't know what.
    #myfloat = 'float64' 
    myfloat = 'float32'

    self.ntrack = self.target_ids.size

    print "Tracking {} particles...".format( self.ntrack )

    ptrack = {
      'redshift':np.zeros( nsnap, dtype=myfloat ), 
      'snum':np.zeros( nsnap, dtype='int16' ),
      'id':np.zeros( self.ntrack, dtype='int64' ), 
      'Ptype':np.zeros( self.ntrack, dtype=('int8',(nsnap,)) ),
      'rho':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ), 
      'sfr':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'T':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'z':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'm':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,)) ),
      'p':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,3)) ),
      'v':np.zeros( self.ntrack, dtype=(myfloat,(nsnap,3)) ), 
    }

    if self.host_halo:
      ptrack['HaloID'] = np.zeros(self.ntrack,dtype=('int32',(nsnap,)))
      ptrack['SubHaloID'] = np.zeros(self.ntrack,dtype=('int32',(nsnap,)))

    ptrack['id'] = self.target_ids
    if self.target_child_ids is not None:
      ptrack['child_id'] = self.target_child_ids

    j = 0

    for snum in self.snaps:

      time_1 = time.time()

      id_finder = IDFinder()
      dfid, redshift, self.attrs = id_finder.find_ids( self.kwargs['sdir'], snum, self.kwargs['types'], self.target_ids, \
                                           target_child_ids=self.target_child_ids, host_halo=self.host_halo )

      ptrack['redshift'][j] = redshift
      ptrack['snum'][j] = snum
      ptrack['Ptype'][:,j] = dfid['Ptype'].values
      ptrack['rho'][:,j] = dfid['rho'].values                                                           # cm^(-3)
      ptrack['sfr'][:,j] = dfid['sfr'].values                                                           # Msun / year   (stellar Age in Myr for star particles)
      ptrack['T'][:,j] = dfid['T'].values                                                               # Kelvin
      ptrack['z'][:,j] = dfid['z'].values                                                               # Zsun (metal mass fraction in Solar units)
      ptrack['m'][:,j] = dfid['m'].values                                                               # Msun (particle mass in solar masses)
      ptrack['p'][:,j,:] = np.array( [ dfid['x0'].values, dfid['x1'].values, dfid['x2'].values ] ).T    # kpc (physical)
      ptrack['v'][:,j,:] = np.array( [ dfid['v0'].values, dfid['v1'].values, dfid['v2'].values ] ).T    # km/s (peculiar - need to add H(a)*r contribution)

      if self.host_halo:
        ptrack['HaloID'][:,j] = dfid['HaloID'].values
        ptrack['SubHaloID'][:,j] = dfid['SubHaloID'].values

      j += 1

      gc.collect()          # helps stop leaking memory ?
      time_2 = time.time()

      # Print output information.
      print 'Snapshot {:>3} | redshift {:>7.3g} | done in {:.3g} seconds'.format(  snum, redshift, time_2 - time_1 )
      sys.stdout.flush()

    return ptrack

  ########################################################################

  def write_tracked_data( self ):
    '''Write tracks to a file.'''

    # Make sure the output location exists
    if not os.path.exists( self.kwargs['outdir'] ):
      os.mkdir( self.kwargs['outdir'] )

    self.outname = 'ptrack_{}.hdf5'.format( self.kwargs['tag'] )

    outpath =  os.path.join( self.kwargs['outdir'], self.outname )

    f = h5py.File( outpath, 'w' )

    # Save data
    for keyname in self.ptrack.keys():
        f.create_dataset( keyname, data=self.ptrack[keyname] )

    # Save the attributes
    for key in self.attrs.keys():
      f.attrs[key] = self.attrs[key]

    # Save the data parameters (kwargs) to the attributes too.
    for key in self.kwargs.keys():
      f.attrs[key] = self.kwargs[key]

    f.close()

########################################################################
########################################################################

class IDFinder( object ):
  '''Finds target ids in a single snapshot.'''

  def __init__( self ):
    pass

  ########################################################################

  def find_ids( self, sdir, snum, types, target_ids, target_child_ids=None, host_halo=0 ):
    '''Find the information for particular IDs in a given snapshot, ordered by the ID list you pass.

    Args:
      sdir (str): The targeted simulation directory.
      snum (int): The snapshot to find the IDs for.
      types (list of ints): Which particle types to target.
      target_ids (np.array): The particle IDs you want to find.
      host_halo (bool): Whether or not to include the host halo in the returned data. (Calculated via AHF)

    Returns:
      dfid (pandas.DataFrame): Dataframe for the selected IDs, ordered by target_ids.
          Contains standard particle information, e.g. position, metallicity, etc.
      redshift (float): Redshift of the snapshot.
      attrs (dict): Dictionary of attributes of the snapshot
    '''

    # Store the target ids for easy access.
    self.sdir = sdir
    self.snum = snum
    self.types = types
    self.target_ids = target_ids
    if target_child_ids is not None:
      self.target_child_ids = target_child_ids

    # Make a big list of the relevant particle data, across all the particle data types
    self.concatenate_particle_data()

    # Find target_ids 
    self.dfid = self.select_ids()

    # Add galaxy and halo data
    if host_halo:
      self.add_environment_data()

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

    full_snap_data = {
      'id' : [],
      'Ptype' : [],
      'rho' : [],
      'sfr' : [],
      'T' : [],
      'z' : [],
      'm' : [],
      'x0' : [],
      'x1' : [],
      'x2' : [],
      'v0' : [],
      'v1' : [],
      'v2' : [],
      }

    time_start = time.time()

    if hasattr( self, 'target_child_ids' ):
      load_additional_ids = True
      full_snap_data['child_id'] = []
    else:
      load_additional_ids = False

    for p_type in self.types:

      P = readsnap.readsnap( self.sdir, self.snum, p_type, load_additional_ids=load_additional_ids, cosmological=1, skip_bh=1, header_only=0)

      if P['k'] < 0:
         continue
      pnum = P['id'].size

      if verbose:
        print '       ...  ', pnum, '   type', p_type , ' particles'

      if 'rho' in P:
          rho = P['rho']
      else:
          rho = [0.,]*pnum

      if 'sfr' in P:
          sfr = P['sfr']
      else:
          sfr = [0.,]*pnum

      if 'u' in P:
          T = readsnap.gas_temperature(P['u'],P['ne'])
      else:
          T = [0.,]*pnum

      thistype = np.zeros(pnum,dtype='int8')
      thistype.fill(p_type)

      full_snap_data['id'].append( P['id'] )
      full_snap_data['Ptype'].append( thistype )
      full_snap_data['rho'].append( rho )
      full_snap_data['sfr'].append( sfr )
      full_snap_data['T'].append( T )
      full_snap_data['z'].append( P['z'][:,0] )
      full_snap_data['m'].append( P['m'] )
      full_snap_data['x0'].append( P['p'][:,0] )
      full_snap_data['x1'].append( P['p'][:,1] )
      full_snap_data['x2'].append( P['p'][:,2] )
      full_snap_data['v0'].append( P['v'][:,0] )
      full_snap_data['v1'].append( P['v'][:,1] )
      full_snap_data['v2'].append( P['v'][:,2] )

      if hasattr( self, 'target_child_ids' ):
        full_snap_data['child_id'].append( P['child_id'] )

    time_end = time.time()

    if verbose:
      print 'readsnap done in ... {:.3g} seconds'.format( time_end - time_start )

    # Convert to numpy arrays
    for key in full_snap_data.keys():
      full_snap_data[key] = np.concatenate( full_snap_data[key] )

    # Save the redshift
    self.redshift = P['redshift']

    # Store the attributes to be used in the final data file.
    attrs_keys = [ 'omega_matter', 'omega_lambda', 'hubble' ]
    self.attrs = {}
    for key in attrs_keys:
      self.attrs[key] = P[key]

    self.full_snap_data = full_snap_data

  ########################################################################

  def select_ids(self):
    '''Function for selecting the targeted ids from the snapshot.'''

    # Setup the index and the way to select the targeted ids
    if hasattr( self, 'target_child_ids' ):
      index = [ self.full_snap_data['id'], self.full_snap_data['child_id'] ]
      target_selection = list( zip( *[ self.target_ids, self.target_child_ids ] ) )
    else:
      index = self.full_snap_data['id']
      target_selection = self.target_ids

    # Make a data frame.
    df = pd.DataFrame( data=self.full_snap_data, index=index )      

    # Sort for faster indexing
    df.sort()

    # Make a data frame selecting only the target ids
    dfid = df.loc[ target_selection ]

    return dfid

  ########################################################################

  def add_environment_data(self):

    AHFile = glob( sdir + '/snap' + g.snap_ext(snum) + 'Rpep..z*.AHF_particles')[0]

    print '\nreading... ', AHFile

    data = read_ahf_particles(AHFile)

    hf = pd.DataFrame( data=data, index=data['id'] )

    # WARNING:  order in hfid may **NOT** be the same as order in theIDS if there are reated indexes in hf !!!  (might be version dependent...)
    hfid = hf.ix[target_ids].copy()             

    del data, hf

   ### NOTE: use keep='first' ??

    # this keeps ONLY the FIRST value corresponding to each repeated id (i.e. now there are NO repeated IDs in hfid_host)
    hfid_host = hfid.drop_duplicates('id', take_last=False, inplace=False)   
    # this maintains the same order as target_ids because now hfid has NO repeated indexes (even if hfid has a different order)
    dfid = dfid.join( hfid_host['HaloID'] )      
    if not np.array_equal( target_ids, dfid['id'].values):
      print 'WARNING!  issue with IDs (1)  !!!'

    # this keeps ALL values EXCEPT the FIRST one for each repeated id (i.e. there might be repeated IDs in hfid_subhost)
    hfid_subhost = hfid[ hfid.duplicated('id', take_last=False) ].copy()  
    # this keeps ONLY the FIRST value corresponding to each repeated id (i.e. now there are NO repeated IDs in hfid_subhost)
    hfid_subhost.drop_duplicates('id', take_last=False, inplace=True)    
    hfid_subhost.rename(columns={'HaloID':'SubHaloID'}, inplace=True)
    dfid = dfid.join( hfid_subhost['SubHaloID'] )
    if not np.array_equal( target_ids, dfid['id'].values):
      print 'WARNING!  issue with IDs (2)  !!!'

    dfid['HaloID'][ pd.isnull(dfid['HaloID']) ] = -1
    dfid['SubHaloID'][ pd.isnull(dfid['SubHaloID']) ] = -1

########################################################################

def read_ahf_particles(filename):
  '''Read a *.AHF_particles file. See http://popia.ft.uam.es/AHF/files/AHF.pdf,
  pg 171 for documentation for such a file.

  Args:
    filename (str): The full file path.
  '''

  time_start = time.time()

  # Get the number of lines
  f = open(filename, "r")
  Nhalos = int( f.readline() )
  f.close()

  df = pd.read_csv(filename, delim_whitespace=True, names=['id','Ptype'], header=0)

  Nlines = df['id'].size
  Ndata = Nlines - Nhalos

  data = { 'id':np.zeros(Ndata,dtype='int64'), 'Ptype':np.zeros(Ndata,dtype='int8'), 'HaloID':np.zeros(Ndata,dtype='int32') }


  jbeg = 0

  for i in xrange(0, Nhalos):

    jend = jbeg + df['id'][ jbeg+i ]

    data['id'][ jbeg : jend ] = df['id'][ jbeg+i+1 : jend+i+1 ].values
    data['Ptype'][ jbeg : jend ] = df['Ptype'][ jbeg+i+1 : jend+i+1 ].values
    data['HaloID'][ jbeg : jend ] = df['Ptype'][ jbeg+i ]

    if i in [1,10,100,1000,10000,100000]:
      print '       HaloID =', df['Ptype'][ jbeg+i ], '   i = ', i

    jbeg = jend

  time_end = time.time()    

  print 'read_ahf_particles done in ... ', time_end - time_start, ' seconds'

  return data

