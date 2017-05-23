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

def find_ids( simdir, snapnum, theTypes, theIDs, HostGalaxy=0, HostHalo=0 ):
  '''Find the information for particular IDs in a given snapshot, ordered by the ID list you pass.

  Args:
    simdir (str): The targeted simulation directory.
    snapnum (int): The snapshot to find the IDs for.
    theTypes (list of ints): Which particle types to target.
    theIDs (np.array): The particle IDs you want to find.
    HostGalaxy (bool): Whether or not to include the host galaxy in the returned data. (Calculated via SKID)
    HostHalo (bool): Whether or not to include the host halo in the returned data. (Calculated via AHF)

  Returns:
    dfid (pandas.DataFrame): Dataframe for the selected IDs, ordered by theIDs.
        Contains standard particle information, e.g. position, metallicity, etc.
    redshift (float): Redshift of the snapshot.
  '''

   snapdir = simdir + '/snapdir_' + g.snap_ext(snapnum)

   print '\nreading... ', snapdir

   id_all    = np.array([],dtype='int64')
   Ptype_all = np.array([],dtype='int8')
   rho_all   = np.array([])
   T_all     = np.array([])
   z_all     = np.array([])
   m_all     = np.array([])
   x0_all    = np.array([])
   x1_all    = np.array([])
   x2_all    = np.array([])
   v0_all    = np.array([])
   v1_all    = np.array([])
   v2_all    = np.array([])

   ###########################
    ### GET PARTICLE DATA ###
   ###########################

   time_start = time.time()

   for Ptype in theTypes:

      P = readsnap(simdir,snapnum,Ptype,cosmological=1,skip_bh=1,header_only=0)

      if P['k'] < 0:
         continue
      pnum = P['id'].size

      print '       ...  ', pnum, '   type', Ptype , ' particles'

      if 'rho' in P:
          rho = P['rho']
      else:
          rho = np.zeros(pnum)    #; rho.fill(-1)

      if 'u' in P:
          T = g.gas_temperature(P['u'],P['ne'])
      else:
          T = np.zeros(pnum)      #; T.fill(-1)

      thistype = np.zeros(pnum,dtype='int8'); thistype.fill(Ptype)

      id_all    = np.append( id_all, P['id'] )
      Ptype_all = np.append( Ptype_all, thistype )
      rho_all   = np.append( rho_all, rho )
      T_all     = np.append( T_all, T )
      z_all     = np.append( z_all, P['z'][:,0] )
      m_all    = np.append( m_all, P['m'] )
      x0_all    = np.append( x0_all, P['p'][:,0] )
      x1_all    = np.append( x1_all, P['p'][:,1] )
      x2_all    = np.append( x2_all, P['p'][:,2] )
      v0_all    = np.append( v0_all, P['v'][:,0] )
      v1_all    = np.append( v1_all, P['v'][:,1] )
      v2_all    = np.append( v2_all, P['v'][:,2] )

      redshift = P['redshift']

   time_end = time.time()

   print 'readsnap done in ... ', time_end - time_start, ' seconds'

   #####################
    ### FIND theIDs ###
   #####################

   allvar = { 'id':id_all, 'Ptype':Ptype_all, 'rho':rho_all, 'T':T_all, 'z':z_all, 'm':m_all, \
              'x0':x0_all, 'x1':x1_all, 'x2':x2_all, \
              'v0':v0_all, 'v1':v1_all, 'v2':v2_all }

   # Make a data frame.
   df = pd.DataFrame( data=allvar, index=id_all )      

   # Make a data frame selecting only the ids.
   dfid = df.ix[theIDs].copy()                         # order in dfid is the same as order in theIDs **IF** there are no repeated indexes in df !!  (version dependent?)

   assert np.array_equal( theIDs, dfid['id'].values):
     print 'WARNING!  issue with IDs (0)  !!!'

   del P, allvar, df                                  



   if (HostGalaxy==1):

      data = read_skid( simdir, snapnum, FileType='grp' )

      data['GalID'][ data['GalID'] == 0 ] = -1                          # redefine No host galaxy as GalID = -1 rather than 0

      gf = pd.DataFrame( data=data, index=data['id'] )

      gf.drop_duplicates('id', take_last=False, inplace=True)           # make sure there are no duplicate ids

      gfid = gf.ix[theIDs].copy()                                       # order in gfid should be the same as order in theIDS 

      del data, gf

      dfid = dfid.join( gfid['GalID'] )                                 # this should maintain the same order as theIDs
      if not np.array_equal( theIDs, dfid['id'].values):
        print 'WARNING!  issue with IDs (0)  !!!'

      #dfid['GalID'][ pd.isnull(dfid['GalID']) ] = -1                   # there shouldn't be any null values



   if (HostHalo==1):

      AHFile = glob( simdir + '/snap' + g.snap_ext(snapnum) + 'Rpep..z*.AHF_particles')[0]

      print '\nreading... ', AHFile

      data = read_AHF_particles(AHFile)

      hf = pd.DataFrame( data=data, index=data['id'] )

      hfid = hf.ix[theIDs].copy()             # WARNING:  order in hfid may **NOT** be the same as order in theIDS if there are reated indexes in hf !!!  (might be version dependent...)
 
      del data, hf

     ### NOTE: use keep='first' ??

      hfid_host = hfid.drop_duplicates('id', take_last=False, inplace=False)   # this keeps ONLY the FIRST value corresponding to each repeated id (i.e. now there are NO repeated IDs in hfid_host)
      dfid = dfid.join( hfid_host['HaloID'] )      # this maintains the same order as theIDs because now hfid has NO repeated indexes (even if hfid has a different order)
      if not np.array_equal( theIDs, dfid['id'].values):
        print 'WARNING!  issue with IDs (1)  !!!'

      hfid_subhost = hfid[ hfid.duplicated('id', take_last=False) ].copy()  # this keeps ALL values EXCEPT the FIRST one for each repeated id (i.e. there might be repeated IDs in hfid_subhost)
      hfid_subhost.drop_duplicates('id', take_last=False, inplace=True)    # this keeps ONLY the FIRST value corresponding to each repeated id (i.e. now there are NO repeated IDs in hfid_subhost)
      hfid_subhost.rename(columns={'HaloID':'SubHaloID'}, inplace=True)
      dfid = dfid.join( hfid_subhost['SubHaloID'] )
      if not np.array_equal( theIDs, dfid['id'].values):
        print 'WARNING!  issue with IDs (2)  !!!'

      dfid['HaloID'][ pd.isnull(dfid['HaloID']) ] = -1
      dfid['SubHaloID'][ pd.isnull(dfid['SubHaloID']) ] = -1
      

   return dfid, redshift
