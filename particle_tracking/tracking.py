#!/usr/bin/env python
'''Tools for tracking particles.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import pandas as pd
import time

import readsnap

########################################################################

class IDFinder(object):

  def __init__(self):
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
    '''

    # Store the targeted ids for easy access.
    self.sdir = sdir
    self.snum = snum
    self.types = types
    self.target_ids = target_ids
    if target_child_ids is not None:
      self.target_child_ids = target_child_ids

    # Make a big list of the relevant particle data, across all the particle data types
    self.full_snap_data = self.concatenate_particle_data()

    # Find target_ids 
    self.dfid = self.select_ids()

    # Add galaxy and halo data
    if host_halo:
      self.add_environment_data()

    return self.dfid, self.redshift

  ########################################################################

  def concatenate_particle_data(self):
    '''Get all the particle data for the snapshot in one big array. (Daniel must use this format somehow...)

    Returns:
      full_snap_data (dict): A dictionary of the concatenated particle data.
    '''

    print 'Reading data...'


    full_snap_data = {
      'id' : [],
      'p_type' : [],
      'rho' : [],
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

      print '       ...  ', pnum, '   type', p_type , ' particles'

      if 'rho' in P:
          rho = P['rho']
      else:
          rho = [0,]*pnum

      if 'u' in P:
          T = readsnap.gas_temperature(P['u'],P['ne'])
      else:
          T = [0,]*pnum

      thistype = np.zeros(pnum,dtype='int8'); thistype.fill(p_type)

      full_snap_data['id'].append( P['id'] )
      full_snap_data['p_type'].append( thistype )
      full_snap_data['rho'].append( rho )
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

    print 'readsnap done in ... ', time_end - time_start, ' seconds'

    # Convert to numpy arrays
    for key in full_snap_data.keys():
      full_snap_data[key] = np.array( full_snap_data[key] ).flatten()

    # Save the redshift
    self.redshift = P['redshift']

    return full_snap_data

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

  def save_targeted_particles( self ):

    pass
    
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

