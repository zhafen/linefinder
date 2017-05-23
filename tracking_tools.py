#!/usr/bin/env python
'''Tools for tracking particles.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import pandas as pd
import time

from readsnap import readsnap

########################################################################

class IDFinder(object):

  def __init__(self):
    pass

  ########################################################################

  def find_ids( self, sdir, snum, types, targeted_ids, host_galaxy=0, host_halo=0 ):
    '''Find the information for particular IDs in a given snapshot, ordered by the ID list you pass.

    Args:
      sdir (str): The targeted simulation directory.
      snum (int): The snapshot to find the IDs for.
      types (list of ints): Which particle types to target.
      targeted_ids (np.array): The particle IDs you want to find.
      host_galaxy (bool): Whether or not to include the host galaxy in the returned data. (Calculated via SKID)
      host_halo (bool): Whether or not to include the host halo in the returned data. (Calculated via AHF)

    Returns:
      dfid (pandas.DataFrame): Dataframe for the selected IDs, ordered by targeted_ids.
          Contains standard particle information, e.g. position, metallicity, etc.
      redshift (float): Redshift of the snapshot.
    '''

    # Store the targeted ids for easy access.
    self.sdir = sdir
    self.snum = snum
    self.types = types
    self.targeted_ids = targeted_ids

    # Make a big list of the relevant particle data, across all the particle data types
    self.full_snap_data = self.concatenate_particle_data()

    # Find targeted_ids 
    self.dfid = self.select_ids()

    # Add galaxy and halo data
    self.add_environment_data( host_galaxy, host_halo )

    return dfid, redshift

  ########################################################################

  def concatenate_particle_data(self):
    '''Get all the particle data for the snapshot in one big array. (Daniel must use this format somehow...)

    Returns:
      full_snap_data (dict): A dictionary of the concatenated particle data.
    '''

    print 'Reading data...'

    id_all    = []
    p_type_all = []
    rho_all   = []
    T_all     = []
    z_all     = []
    m_all     = []
    x0_all    = []
    x1_all    = []
    x2_all    = []
    v0_all    = []
    v1_all    = []
    v2_all    = []

    time_start = time.time()

    for p_type in types:

      P = readsnap( self.sdir, self.snum, p_type, cosmological=1, skip_bh=1, header_only=0)

      if P['k'] < 0:
         continue
      pnum = P['id'].size

      print '       ...  ', pnum, '   type', p_type , ' particles'

      if 'rho' in P:
          rho = P['rho']
      else:
          rho = [0,]*pnum    #; rho.fill(-1)

      if 'u' in P:
          T = g.gas_temperature(P['u'],P['ne'])
      else:
          T = [0,]*pnum      #; T.fill(-1)

      thistype = np.zeros(pnum,dtype='int8'); thistype.fill(p_type)

      id_all.append( P['id'] )
      p_type_all.append( thistype )
      rho_all.append( rho )
      T_all.append( T )
      z_all.append( P['z'][:,0] )
      m_all.append( P['m'] )
      x0_all.append( P['p'][:,0] )
      x1_all.append( P['p'][:,1] )
      x2_all.append( P['p'][:,2] )
      v0_all.append( P['v'][:,0] )
      v1_all.append( P['v'][:,1] )
      v2_all.append( P['v'][:,2] )

      redshift = P['redshift']

    time_end = time.time()

    print 'readsnap done in ... ', time_end - time_start, ' seconds'

    full_snap_data = { 'id':id_all, 'p_type':p_type_all, 'rho':rho_all, 'T':T_all, 'z':z_all, 'm':m_all, \
              'x0':x0_all, 'x1':x1_all, 'x2':x2_all, \
              'v0':v0_all, 'v1':v1_all, 'v2':v2_all }

    # Convert to numpy arrays
    for key in full_snap_data.keys():
      full_snap_data[key] = np.array( full_snap_data[key] )

    return full_snap_data

  ########################################################################

  def select_ids(self):

    # Make a data frame.
    df = pd.DataFrame( data=self.full_snap_data, index=self.full_snap_data['id'] )      

    # Make a data frame selecting only the ids.
    # order in dfid is the same as order in targeted_ids **IF** there are no repeated indexes in df !!  (version dependent?)
    dfid = df.ix[targeted_ids].copy()

    assert np.array_equal( self.targeted_ids, dfid['id'].values)

    return dfid

  ########################################################################

  def add_environment_data(self):

    if (host_galaxy==1):

      data = read_skid( sdir, snum, FileType='grp' )

      # redefine No host galaxy as GalID = -1 rather than 0
      data['GalID'][ data['GalID'] == 0 ] = -1                          

      gf = pd.DataFrame( data=data, index=data['id'] )

      # make sure there are no duplicate ids
      gf.drop_duplicates('id', take_last=False, inplace=True)           

      # order in gfid should be the same as order in theIDS 
      gfid = gf.ix[targeted_ids].copy()                                       

      del data, gf

      # this should maintain the same order as targeted_ids
      dfid = dfid.join( gfid['GalID'] )                                 
      if not np.array_equal( targeted_ids, dfid['id'].values):
        print 'WARNING!  issue with IDs (0)  !!!'

      # there shouldn't be any null values
      #dfid['GalID'][ pd.isnull(dfid['GalID']) ] = -1                   

    if (host_halo==1):

      AHFile = glob( sdir + '/snap' + g.snap_ext(snum) + 'Rpep..z*.AHF_particles')[0]

      print '\nreading... ', AHFile

      data = read_AHF_particles(AHFile)

      hf = pd.DataFrame( data=data, index=data['id'] )

      # WARNING:  order in hfid may **NOT** be the same as order in theIDS if there are reated indexes in hf !!!  (might be version dependent...)
      hfid = hf.ix[targeted_ids].copy()             

      del data, hf

     ### NOTE: use keep='first' ??

      # this keeps ONLY the FIRST value corresponding to each repeated id (i.e. now there are NO repeated IDs in hfid_host)
      hfid_host = hfid.drop_duplicates('id', take_last=False, inplace=False)   
      # this maintains the same order as targeted_ids because now hfid has NO repeated indexes (even if hfid has a different order)
      dfid = dfid.join( hfid_host['HaloID'] )      
      if not np.array_equal( targeted_ids, dfid['id'].values):
        print 'WARNING!  issue with IDs (1)  !!!'

      # this keeps ALL values EXCEPT the FIRST one for each repeated id (i.e. there might be repeated IDs in hfid_subhost)
      hfid_subhost = hfid[ hfid.duplicated('id', take_last=False) ].copy()  
      # this keeps ONLY the FIRST value corresponding to each repeated id (i.e. now there are NO repeated IDs in hfid_subhost)
      hfid_subhost.drop_duplicates('id', take_last=False, inplace=True)    
      hfid_subhost.rename(columns={'HaloID':'SubHaloID'}, inplace=True)
      dfid = dfid.join( hfid_subhost['SubHaloID'] )
      if not np.array_equal( targeted_ids, dfid['id'].values):
        print 'WARNING!  issue with IDs (2)  !!!'

      dfid['HaloID'][ pd.isnull(dfid['HaloID']) ] = -1
      dfid['SubHaloID'][ pd.isnull(dfid['SubHaloID']) ] = -1
      

