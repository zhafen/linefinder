import numpy as np
import scipy.interpolate
import scipy.ndimage
import math
import ctypes
import gadget as g
import utilities as util
import os as os
import sys as sys
import h5py as h5py
import pandas as pd
from glob import glob
import string as string
import time as time
from daa_constants import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
#import gc


def matchgal( simdir_1,                  # feedback
              simdir_2,                  # no-feedback
              snapnum,
              StarMassMin=1e10 ):

   # --- define GROUP LIST ---
   gal_1 = read_skid( simdir_1, snapnum, FileType='stat' )
   ind_gal = np.where( gal_1['StarMass'] >= StarMassMin )[0]
   ngroup = ind_gal.size

   if ngroup == 0:
     return None

   matchgal = { 'GalID':np.zeros( [ ngroup, 2 ], dtype='i' ),
                'nstar':np.zeros( [ ngroup, 2 ], dtype='i' ),
                'simdir':['sim1','sim2'],
                'redshift':-1. }

   matchgal['simdir'][0] = simdir_1
   matchgal['simdir'][1] = simdir_2
   matchgal['GalID'][:,0] = gal_1['GalID'][ind_gal]
   matchgal['GalID'][:,1] = -1


   # --- sim 1 ---
   hdr = g.readsnap( simdir_1, snapnum, 0, header_only=1)
   matchgal['redshift'] = hdr['redshift']
   nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
   nstar = hdr['npartTotal'][4]
   tmp = read_skid( simdir_1, snapnum, FileType='grp' )
   grp_1 = { 'id':tmp['id'][ nskip : nskip+nstar ],
             'GalID':tmp['GalID'][ nskip : nskip+nstar ] }
 
   # --- sim 2 ---
   gal_2 = read_skid( simdir_2, snapnum, FileType='stat' )
   hdr = g.readsnap( simdir_2, snapnum, 0, header_only=1)
   nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
   nstar = hdr['npartTotal'][4]
   tmp = read_skid( simdir_2, snapnum, FileType='grp' )
   grp_2 = { 'id':tmp['id'][ nskip : nskip+nstar ],
             'GalID':tmp['GalID'][ nskip : nskip+nstar ] }

   # --- find matching groups ---
   for ng in range(ngroup):

      ind_1 = np.where( grp_1['GalID'] == matchgal['GalID'][ng,0] )[0]
      if ind_1.size == 0:
         continue
      df = pd.DataFrame( data=grp_2, index=grp_2['id'] )
      df = df['GalID'].ix[grp_1['id'][ind_1]].dropna()
      df = df.iloc[df.nonzero()[0]]
      if (df.size > 32) and (df.mode().size) > 0:
         matchgal['GalID'][ng,1] = df.mode().values[0]

         ind_2 = np.where( grp_2['GalID'] == matchgal['GalID'][ng,1] )[0]
         if ind_2.size == 0:
            matchgal['GalID'][ng,1] = -1                  # not valid
            continue
         df = pd.DataFrame( data=grp_1, index=grp_1['id'] )
         df = df['GalID'].ix[grp_2['id'][ind_2]].dropna()
         df = df.iloc[df.nonzero()[0]]
         if (df.size < 32) or (df.mode().size) == 0:
            matchgal['GalID'][ng,1] = -1                  # not valid
            continue
         if df.mode().values[0] != matchgal['GalID'][ng,0]:
            matchgal['GalID'][ng,1] = -1                  # not valid
            continue
            
         matchgal['nstar'][ng,0] = gal_1['nstar'][ matchgal['GalID'][ng,0] -1 ]
         if matchgal['GalID'][ng,1] > 0:
           matchgal['nstar'][ng,1] = gal_2['nstar'][ matchgal['GalID'][ng,1] -1 ]

         print '\nsnapnum = ', snapnum, ':  ', ng+1, ' / ', ngroup, '      nstar = ', ind_1.size, matchgal['nstar'][ng,0], ' --> ', ind_2.size, matchgal['nstar'][ng,1]
         sys.stdout.flush()


   # --- write output file ---
   outname = 'matchgal_' + simdir_2[simdir_2.find('_')+1:] + '_' + g.snap_ext(snapnum) + '.hdf5'
   outfile = simdir_1 + '/skid/' + outname
   print 'writing file ... ', outfile
   if os.path.isfile(outfile):
     os.remove(outfile)
   f = h5py.File(outfile, 'w')
   for keyname in matchgal.keys():
       f.create_dataset(keyname, data=matchgal[keyname])
   f.close()

   print '\n\nmatchgal done!'
   return None



def progen( simdir, snaplist,
            NstarMin=100 ):

   nsnap = len(snaplist)

   # --- define GROUP LIST ---
   gal = read_skid( simdir, snaplist[0], FileType='stat' )
   ind_gal = np.where( gal['nstar'] >= NstarMin )[0]
   ngroup = ind_gal.size
   #ngroup = gal['GalID'].size

   progen = { 'GalID':np.zeros( [ ngroup, nsnap ], dtype='i' ),
              'nstar':np.zeros( [ ngroup, nsnap ], dtype='i' ),
              'snapnum':np.zeros( nsnap, dtype='i' ),
              'redshift':np.zeros( nsnap, dtype='f' ) }

   progen['GalID'][:,0] = gal['GalID'][ind_gal]
   progen['GalID'][:,1:] = -1

   progen['snapnum'][:] = -1
   progen['redshift'][:] = -1


   skip_fill = 0

   for i in range(nsnap-1):

      # --- CURRENT GROUPS ---
      hdr = g.readsnap( simdir, snaplist[i], 0, header_only=1)

      progen['snapnum'][i] = snaplist[i]
      progen['redshift'][i] = hdr['redshift']

      nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
      nstar = hdr['npartTotal'][4]
      tmp = read_skid( simdir, snaplist[i], FileType='grp' )
      grp = { 'id':tmp['id'][ nskip : nskip+nstar ],
              'GalID':tmp['GalID'][ nskip : nskip+nstar ] }

      # --- PROGENITOR GROUPS ---
      hdr = g.readsnap( simdir, snaplist[i+1], 0, header_only=1)
      nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
      nstar = hdr['npartTotal'][4]
      if nstar == 0:
         skip_fill = 1
         break
      tmp = read_skid( simdir, snaplist[i+1], FileType='grp' )
      grp_pro = { 'id':tmp['id'][ nskip : nskip+nstar ],
                  'GalID':tmp['GalID'][ nskip : nskip+nstar ] }

      gal_pro = read_skid( simdir, snaplist[i+1], FileType='stat' )
      if gal_pro['redshift'] == -1:
         skip_fill = 1
         break
      #count_pro = np.zeros( gal_pro['GalID'].size + 1 )

      # --- find progenitor groups ---
      for ng in range(ngroup):

         print 'snapnum =', snaplist[i], ':   ', ng+1, ' / ', ngroup
         #count_pro[:] = 0
         sys.stdout.flush()

         ind = np.where( grp['GalID'] == progen['GalID'][ng,i] )[0]
         if ind.size == 0:
            continue
         progen['nstar'][ng,i] = ind.size

         df = pd.DataFrame( data=grp_pro, index=grp_pro['id'] )
         df = df['GalID'].ix[grp['id'][ind]].dropna()
         df = df.iloc[df.nonzero()[0]]
         if df.size == 1:
            progen['GalID'][ng,i+1] = df.values[0]
         elif df.size > 1:
            progen['GalID'][ng,i+1] = df.mode().values[0]


   if skip_fill == 0:
      # --- fill in data for last snapshot
      hdr = g.readsnap( simdir, snaplist[i+1], 0, header_only=1)
      progen['snapnum'][i+1] = snaplist[i+1]
      progen['redshift'][i+1] = hdr['redshift']
      nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
      nstar = hdr['npartTotal'][4]
      tmp = read_skid( simdir, snaplist[i+1], FileType='grp' )
      grp = { 'id':tmp['id'][ nskip : nskip+nstar ],
              'GalID':tmp['GalID'][ nskip : nskip+nstar ] }
      for ng in range(ngroup):
         ind = np.where( grp['GalID'] == progen['GalID'][ng,i+1] )[0]
         if ind.size == 0:
            continue
         progen['nstar'][ng,i+1] = ind.size


   # --- clean up progen structure
   ind = np.where( progen['snapnum'] >= 0 )[0]

   # --- write output file ---
   outfile = simdir + '/skid/progen.hdf5'
   if os.path.isfile(outfile):
     os.remove(outfile)
   f = h5py.File(outfile, 'w')
   for keyname in progen.keys():
       if progen[keyname][:].ndim == 1:
          f.create_dataset(keyname, data=progen[keyname][ind])
       else:
          f.create_dataset(keyname, data=progen[keyname][:,ind])
   f.close()

   print '\n\nprogen done!'
   return progen



"""
def progen( simdir, snaplist, 
            StarMassMin=1e10 ):

   nsnap = len(snaplist)

   # --- define GROUP LIST ---
   gal = read_skid( simdir, snaplist[0], FileType='stat' )
   ind_gal = np.where( gal['StarMass'] >= StarMassMin )[0]
   ngroup = ind_gal.size
   #ngroup = gal['GalID'].size

   progen = { 'GalID':np.zeros( [ ngroup, nsnap ], dtype='i' ),
              'nstar':np.zeros( [ ngroup, nsnap ], dtype='i' ),
              'snapnum':np.zeros( nsnap, dtype='i' ),
              'redshift':np.zeros( nsnap, dtype='f' ) }

   progen['GalID'][:,0] = gal['GalID'][ind_gal]
   progen['GalID'][:,1:] = -1


   for i in range(nsnap-1):

      # --- CURRENT GROUPS ---
      hdr = g.readsnap( simdir, snaplist[i], 0, header_only=1)

      progen['snapnum'][i] = snaplist[i]
      progen['redshift'][i] = hdr['redshift']

      nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
      nstar = hdr['npartTotal'][4]
      tmp = read_skid( simdir, snaplist[i], FileType='grp' )
      grp = { 'id':tmp['id'][ nskip : nskip+nstar ],
              'GalID':tmp['GalID'][ nskip : nskip+nstar ] }

      # --- PROGENITOR GROUPS ---
      hdr = g.readsnap( simdir, snaplist[i+1], 0, header_only=1)
      nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
      nstar = hdr['npartTotal'][4]
      if nstar == 0:
         break
      tmp = read_skid( simdir, snaplist[i+1], FileType='grp' )
      grp_pro = { 'id':tmp['id'][ nskip : nskip+nstar ],
                  'GalID':tmp['GalID'][ nskip : nskip+nstar ] }

      #gal_pro = read_skid( simdir, snaplist[i+1], FileType='stat' )
      #count_pro = np.zeros( gal_pro['GalID'].size + 1 )

      # --- find progenitor groups ---
      for ng in range(ngroup):

         print 'snapnum =', snaplist[i], ':   ', ng+1, ' / ', ngroup
         #count_pro[:] = 0
         sys.stdout.flush()

         ind = np.where( grp['GalID'] == progen['GalID'][ng,i] )[0]
         if ind.size == 0:
            continue
         progen['nstar'][ng,i] = ind.size

         df = pd.DataFrame( data=grp_pro, index=grp_pro['id'] )
         df = df['GalID'].ix[grp['id'][ind]].dropna()
         df = df.iloc[df.nonzero()[0]]
         if df.size == 1:
            progen['GalID'][ng,i+1] = df.values[0]
         elif df.size > 1:
            progen['GalID'][ng,i+1] = df.mode().values[0]

   # --- fill in data for last snapshot
   hdr = g.readsnap( simdir, snaplist[i+1], 0, header_only=1)
   progen['snapnum'][i+1] = snaplist[i+1]
   progen['redshift'][i+1] = hdr['redshift']
   nskip = hdr['npartTotal'][0] + hdr['npartTotal'][1]
   nstar = hdr['npartTotal'][4]
   tmp = read_skid( simdir, snaplist[i+1], FileType='grp' )
   grp = { 'id':tmp['id'][ nskip : nskip+nstar ],
           'GalID':tmp['GalID'][ nskip : nskip+nstar ] }
   for ng in range(ngroup):
      ind = np.where( grp['GalID'] == progen['GalID'][ng,i+1] )[0]
      if ind.size == 0:
         continue
      progen['nstar'][ng,i+1] = ind.size
   
   # --- write output file ---
   outfile = simdir + '/skid/progen.hdf5'
   if os.path.isfile(outfile):
     os.remove(outfile)
   f = h5py.File(outfile, 'w')
   for keyname in progen.keys():
       f.create_dataset(keyname, data=progen[keyname])
   f.close()

   print '\n\nprogen done!'
   return None
"""


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


def progen_idlist( simdir ):

   progen = h5py.File( simdir + '/skid/progen.hdf5', 'r')

   snaplist = progen['snapnum'][:]
   nsnap = len(snaplist)

   # --- find most massive galaxy at final snapshot ---
   gal = read_skid( simdir, snaplist[0], FileType='stat' )
   ind = np.where( gal['StarMass'] == np.max(gal['StarMass']) )[0]
   ind = np.where( progen['GalID'][:,0] == gal['GalID'][ind[0]] )[0]

   print '\nFinding particles on GalID = %i,   StarMass = %.4e,   redshift = %.2f' % ( progen['GalID'][ind[0],0], np.max(gal['StarMass']), progen['redshift'][0] )

   # --- find particles in "main" galaxy at ALL times ---
   theIDs = np.array([],dtype='int64')

   for i in range(nsnap):

      grp = read_skid( simdir, snaplist[i], FileType='grp' )
      ind_grpid = np.where( grp['GalID'] == progen['GalID'][ind[0],i] )[0]
      id_all    = np.append( theIDs, grp['id'][ind_grpid] )
      theIDs = np.unique( id_all )
      print '\n', i, ' / ', nsnap, '   npart =', ind_grpid.size, '  Ntot =', theIDs.size

   # --- create output file ---
   outfile = simdir +  '/skid/progen_idlist.hdf5'
   if os.path.isfile(outfile):
     os.remove(outfile)
   f = h5py.File(outfile, 'w')
   f.create_dataset('id', data=theIDs)
   f.close()

   progen.close()

   print '\n\nprogen_idlist done!'
   return None




#########################################################################################################################
#########################################################################################################################
#########################################################################################################################




def select_ids( simdir, snapnum,
                mode='skid', 
                param={ } ):

####################################################################################


   if mode == 'skid':
 
      header = g.readsnap(simdir,snapnum,0,cosmological=1,header_only=1)

      gal = read_skid( simdir, snapnum, FileType='stat' )

      # --- find most massive galaxy
      ind_gal = np.where( gal['StarMass'] == np.max(gal['StarMass']) )[0]
      thegalID = gal['GalID'][ind_gal[0]]

      # --- find particles in most massive galaxy
      grp = read_skid( simdir, snapnum, FileType='grp' )
      ind_grpid = np.where( grp['GalID'] == thegalID )[0]

      if 'Ptype' in param:
         if param['Ptype'] == 0:
            is_gas = ind_grpid < header['npartTotal'][0]
            ind_myids = ind_grpid[ is_gas ]  
         elif param['Ptype'] == 4:
            is_star = ind_grpid >= header['npartTotal'][0] + header['npartTotal'][1]
            ind_myids = ind_grpid[ is_star ]
         else:
            ind_myids = ind_grpid
      else:
         ind_myids = ind_grpid

      if 'nmax' in param:
         if ind_myids.size > param['nmax']:
            np.random.seed(seed=1234)
            np.random.shuffle(ind_myids)
            ind_myids = ind_myids[0:param['nmax']]


      theIDs = np.unique( grp['id'][ind_myids] )    # make sure IDs are unique (and sort them anyway)      

      print '\n  --> selecting  %d  %s  particles at  z = %.3f  from  GalaxyID = %d  (Mstar = %.3e)' % \
            ( theIDs.size, param, header['redshift'], thegalID, gal['StarMass'][ind_gal[0]] )
      print '------------------------------------------------------------------------------------------------\n'

      return theIDs

####################################################################################


   elif mode == 'AHF':                 # WARNING!! need to update!!

      #######################
       ### READ SNAPSHOT ###
      #######################

      P = g.readsnap(simdir,snapnum,Ptype,cosmological=1,skip_bh=1,header_only=0)


      ########################
       ### READ HALO FILE ###
      ########################

      # Halo ID in rank of overdensity at the given redshift (0 = most massive)
      # Host ID = -1 for main halos, = Halo ID of host for subhalos

      hfile = open( simdir + '/halo_00000.dat' )
      htxt = np.loadtxt(hfile)
      hfile.close()

      h_z = htxt[:,0]
      h_Rvir = htxt[:,12]
      h_pos = np.array( [ htxt[:,6], htxt[:,7], htxt[:,8] ] ).T
   
      ascale = 1. / ( 1. + h_z )

      h_Rvir     *= hinv*ascale   # now in physical kpc   (check that Rvir was actually comoving)
      h_pos[:,0] *= hinv*ascale
      h_pos[:,1] *= hinv*ascale
      h_pos[:,2] *= hinv*ascale

      ind_redsh = np.where( np.abs(h_z-P['redshift']) == np.abs(h_z-P['redshift']).min() )[0]


      #################################
       ### CHOOSE TARGET PARTICLES ###
      #################################

      r = P['p'] - h_pos[ind_redsh,:]
      R = np.sqrt((r*r).sum(axis=1))

      ind = np.where( R < 0.1 * h_Rvir[ind_redsh] )[0]
      nind = ind.size

      if nind > ntrack:
         np.random.seed(seed=1234)
         #ind_track = ind[0:ntrack]
         #ind_rand = np.random.choice( nind, ntrack )
         #ind_track = ind[ind_rand]
         np.random.shuffle(ind)
         ind_track = ind[0:ntrack]
      else:
         ind_track = ind

 
      theIDs = np.unique( P['id'][ind_track] )    # make sure IDs are unique (and sort them anyway)

      print '\n  --> selecting ', theIDs.size, ' type =', Ptype, ' particles at  z = ' + '%.3f' % P['redshift'], '  ', h_z[ind_redsh]
      print '------------------------------------------------------------------------------------------------\n'

 
      return theIDs
   

   else:
      print '  ---> what is going on?'
      return None



#########################################################################################################################
#########################################################################################################################
#########################################################################################################################



def find_ids( simdir, snapnum, theTypes, theIDs, 
              HostGalaxy=0, HostHalo=0 ):


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

      #P = g.readsnap(snapdir,snapnum,Ptype,cosmological=1,skip_bh=1,header_only=0)
      P = g.readsnap(simdir,snapnum,Ptype,cosmological=1,skip_bh=1,header_only=0)

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
   #print 'id_all.size = ', id_all.size, np.unique( id_all ).size

   #####################
    ### FIND theIDs ###
   #####################


   allvar = { 'id':id_all, 'Ptype':Ptype_all, 'rho':rho_all, 'T':T_all, 'z':z_all, 'm':m_all, \
              'x0':x0_all, 'x1':x1_all, 'x2':x2_all, \
              'v0':v0_all, 'v1':v1_all, 'v2':v2_all }

   df = pd.DataFrame( data=allvar, index=id_all )      

   dfid = df.ix[theIDs].copy()                         # order in dfid is the same as order in theIDs **IF** there are no repeated indexes in df !!  (version dependent?)

   if not np.array_equal( theIDs, dfid['id'].values):
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


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


def find_haloids(simdir,snapnum,theIDs):

  
   ############################
    ### GET HOST HALO INFO ###
   ############################

   AHFile = glob( simdir + '/snap' + g.snap_ext(snapnum) + 'Rpep..z*.AHF_particles')[0]

   print '\nreading... ', AHFile

   data = read_AHF_particles(AHFile)

   df = pd.DataFrame( data=data, index=data['id'] )

   dfid = df.ix[theIDs]              # WARNING:  order in dfid may **NOT** the same as order in theIDS if there are reated indexes in df !!!  (might be version dependent...)


   # df1.join(df2)    try it !!!

   nIDs = len(theIDs)

   result = { 'id':np.zeros(nIDs,dtype='int64'), 'HaloID':np.zeros(nIDs,dtype='int64'), 'HostHaloID':np.zeros(nIDs,dtype='int64') }

   print '\nassigning host halo IDs...'

   time_start = time.time()

   """
   for i in xrange(0, nIDs): 

      if i % 1e4 == 0:
        print '      ', i
 
      result['id'][i] = theIDs[i]


      indID = np.where( dfid['id'] == theIDs[i] )[0]

      if indID.size == 0:
         result['HaloID'][i] = -1
         result['HostHaloID'][i] = -1
 
      if indID.size == 1:
         result['HaloID'][i] = dfid['HaloID'].values[indID[0]]
         result['HostHaloID'][i] = -1
 
      if indID.size == 2:
         result['HaloID'][i] = dfid['HaloID'].values[indID[1]]
         result['HostHaloID'][i] = dfid['HaloID'].values[indID[0]]

      if indID.size > 2:
         result['HaloID'][i] = dfid['HaloID'].values[indID[-1]]
         result['HostHaloID'][i] = dfid['HaloID'].values[indID[0]]
         print 'hey!!!  more than one sub-halo for id =', theIDs[i]
   """


   dfid.sort('id',inplace=True)     # sort by particle ID
   
   insert = np.searchsorted( dfid['id'].values, theIDs )
   cond = theIDs == dfid['id'].values[insert]
   cond_not = np.logical_not( cond )

   insert_1 = np.copy(insert);  insert_1[insert+1 <= insert.max()] +=1
   cond_1 = cond  &  theIDs == dfid['id'].values[insert_1]

   insert_2 = np.copy(insert_1);  insert_2[insert_1+1 <= insert_1.max()] +=1
   cond_2 = cond_1  &  theIDs == dfid['id'].values[insert_2]


   result['id'][:] = theIDs                                  # still need to check this...

   result['HaloID'][:] = -1
   result['HostHaloID'][:] = -1   

   result['HaloID'][cond] = dfid['HaloID'].values[insert[cond]]

   result['HaloID'][cond_1] = dfid['HaloID'].values[insert_1[cond_1]]
   result['HostHaloID'][cond_1] = dfid['HaloID'].values[insert[cond_1]]

   result['HaloID'][cond_2] = dfid['HaloID'].values[insert_2[cond_2]]   


   time_end = time.time()
   print 'halo assignment done in ... ', time_end - time_start, ' seconds\n'


   return result, dfid




def read_AHF_particles(file):

   time_start = time.time()

   f = open(file, "r")
   Nhalos = int( f.readline() )
   f.close()

   df = pd.read_csv(file, delim_whitespace=True, names=['id','Ptype'], header=0)

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

   print 'read_AHF_particles done in ... ', time_end - time_start, ' seconds'

   return data

##############################################################################################
##############################################################################################


def read_AHF_halos( simdir, snapnum, AHFdir='AHF', readheader=True, h=h ):


   #AHFile = glob( simdir + '/' + AHFdir + '/snap' + g.snap_ext(snapnum) + '*.AHF_halos')[0]
   if len( glob( simdir + '/' + AHFdir + '/snap' + g.snap_ext(snapnum) + '*.AHF_halos') ) == 0:
     AHFdir = 'AHF/%05d/' % snapnum
     AHFile = glob( simdir + '/' + AHFdir + '*.AHF_halos')[0]
   else:
     AHFile = glob( simdir + '/' + AHFdir + '/snap' + g.snap_ext(snapnum) + '*.AHF_halos')[0]

   print '\nReading ' + AHFile

   hfile = open( AHFile )
   htxt = np.loadtxt(hfile)
   hfile.close()

   if readheader:

     header = g.readsnap(simdir,snapnum,0,cosmological=1,skip_bh=1,header_only=1)
     if header['k'] == -1:
        snapname = glob( simdir + '/snap*' + g.snap_ext(snapnum) + '*.hdf5')[0]
        snapname = snapname[ len(simdir)+1 : -9 ]
        header = g.readsnap(simdir,snapnum,0,cosmological=1,skip_bh=1,header_only=1,snapshot_name=snapname)

     redshift = header['redshift']
     h = header['hubble']
     print snapnum, '  z = ', AHFile[-16:-10].split('z')[-1], header['redshift'], '  h =', h

   else:

     redshift = float( AHFile[-16:-10].split('z')[-1] )
     print snapnum, '  z = ', redshift, '  h =', h

   ascale = 1. / ( 1. + redshift )
   hinv = 1./h


   halos = { 'z': redshift,
             'id': htxt[:,0],
             'HostID': htxt[:,1],
             'Mvir': htxt[:,3] * hinv,     # in Msun
             'M_gas': htxt[:,53] * hinv,
             'M_star': htxt[:,73] * hinv,
             'Rvir': htxt[:,11] * hinv * ascale,   # physical kpc
             'cNFW': htxt[:,42],
             'p_mbp': np.array( [ htxt[:,46], htxt[:,47], htxt[:,48] ] ).T * hinv * ascale * 1e3,       # WARNING! now in physical kpc
             'pos': np.array( [ htxt[:,5], htxt[:,6], htxt[:,7] ] ).T * hinv * ascale }           # physical kpc



   print 'Done.\n'

   return halos


##############################################################################################
##############################################################################################


def read_AHF_halos_ALL(simdir, nsnap=0, nhalos=1, AHFdir='AHF', outputlist='snapshot_scale-factors.txt'):

   path = simdir + '/' + AHFdir + '/snap' + '*.AHF_halos'
   print '\nReading ' + path + ' ... [nsnap,nhalos] =', nsnap, nhalos

   AHFiles = glob( path )
   AHFiles.sort()
   AHFiles.reverse()
   if nsnap > 0:
      AHFiles = AHFiles[0:nsnap]

   nsnap = len(AHFiles)

   allhalos = { 'z':np.zeros(nsnap,dtype='float64'),
                'snapnum':np.zeros(nsnap,dtype='int32'),
                'id':np.zeros(nhalos,dtype=('int32',(nsnap,))),
                'HostID':np.zeros(nhalos,dtype=('int32',(nsnap,))),
                'Mvir':np.zeros(nhalos,dtype=('float64',(nsnap,))),
                'Mgas':np.zeros(nhalos,dtype=('float64',(nsnap,))),
                'Mstar':np.zeros(nhalos,dtype=('float64',(nsnap,))),
                'nstar':np.zeros(nhalos,dtype=('int32',(nsnap,))),
                'Rvir':np.zeros(nhalos,dtype=('float64',(nsnap,))),
                'pos':np.zeros(nhalos,dtype=('float64',(nsnap,3))),
                'vel':np.zeros(nhalos,dtype=('float64',(nsnap,3))),
                'p_mbp':np.zeros(nhalos,dtype=('float64',(nsnap,3))) }

   j = 0
   for FilePath in AHFiles:

      allhalos['z'][j] = float( FilePath[-16:-10].split('z')[-1] )  # extract redshift from name

      allhalos['snapnum'][j] = int( FilePath.split('/')[-1][4:7] )  # extract snapnum from name

      hfile = open( FilePath )
      htxt = np.loadtxt(hfile)
      hfile.close()
      if (htxt.size <= 0) or (len(htxt.shape)<2):
        print 'skipping z =', allhalos['z'][j], '  snapnum =', allhalos['snapnum'][j] 
        continue

      nmax = np.min( [ nhalos, htxt[:,0].size ] )

      allhalos['id'][:nmax,j] = htxt[0:nmax,0]

      allhalos['HostID'][:nmax,j] = htxt[0:nmax,1]

      allhalos['Mvir'][:nmax,j] = htxt[0:nmax,3]

      allhalos['Mgas'][:nmax,j] = htxt[0:nmax,53]

      allhalos['Mstar'][:nmax,j] = htxt[0:nmax,73]

      allhalos['nstar'][:nmax,j] = htxt[0:nmax,72]

      allhalos['Rvir'][:nmax,j] = htxt[0:nmax,11]

      allhalos['pos'][:nmax,j,:] = np.array( [ htxt[0:nmax,5], htxt[0:nmax,6], htxt[0:nmax,7] ] ).T

      allhalos['vel'][:nmax,j,:] = np.array( [ htxt[0:nmax,8], htxt[0:nmax,9], htxt[0:nmax,10] ] ).T

      allhalos['p_mbp'][:nmax,j,:] = np.array( [ htxt[0:nmax,46], htxt[0:nmax,47], htxt[0:nmax,48] ] ).T 

      j += 1
 
   header = g.readsnap(simdir,0,0,header_only=1)
   hinv = 1. / header['hubble']

   if len(outputlist) > 0:        # assumes AHF output available for all snapshots
     file = open( simdir + '/' + outputlist )
     a = np.loadtxt(file)[::-1]   # from low to high redshift
     file.close()
     z = 1./a - 1.
     j = np.argmin( np.abs(z-allhalos['z'][0]) )
     if np.min( np.abs(allhalos['z'][:] - z[j:j+nsnap]) ) > 1e-3:
       print 'problem with redshifts!!'
       return
     allhalos['z'][:] = z[j:j+nsnap]
     ascale = a[j:j+nsnap]
   else:
     ascale = 1./(1.+allhalos['z'])

   allhalos['Mvir'] *= hinv          # in Msun
   allhalos['Mgas'] *= hinv
   allhalos['Mstar'] *= hinv
   allhalos['Rvir'] *= hinv*ascale   # physical kpc
   for i in range(3):
      allhalos['pos'][:,:,i] *= hinv*ascale
      allhalos['p_mbp'][:,:,i] *= hinv*ascale*1e3
                                    # peculiar velocity is already in km/s

   print 'Done.\n'

   return allhalos


##############################################################################################
##############################################################################################

def read_halo(simdir, halonum=0, cap=1, smooth=0, AHFdir='AHF', outputlist='snapshot_scale-factors.txt' ):

   ########################
    ### READ HALO FILE ###
   ########################

   # Halo ID in rank of overdensity at the given redshift (0 = most massive)
   # Host ID = -1 for main halos, = Halo ID of host for subhalos
 
   FilePath = simdir + '/' + AHFdir + '/' + 'halo_%05d.dat' % halonum
   print '\nReading ' + FilePath + ' ... [cap,smooth] =', cap, smooth
   hfile = open( FilePath )
   htxt = np.loadtxt(hfile)
   hfile.close()

   # check for redshift order (it should go from low-z to high-z)
   if htxt[0,0] > htxt[-1,0]:
     htxt = htxt[::-1,:]

   # check for last entry repeated:
   if np.abs(htxt[-1,0]-htxt[-2,0]) < 1e-4:
     htxt = htxt[:-1,:]

   halo = { 'z':htxt[:,0],                    # this is only approximate redshift from AHF file names!
            'id':htxt[:,1].astype(int), 
            'Mvir':htxt[:,4], 
            'Rvir':htxt[:,12], 
            'pos':np.array( [ htxt[:,6], htxt[:,7], htxt[:,8] ] ).T, 
            'vel':np.array( [ htxt[:,9], htxt[:,10], htxt[:,11] ] ).T,
            'p_mbp': np.array( [ htxt[:,47], htxt[:,48], htxt[:,49] ] ).T,
            'v_mbp': np.array( [ htxt[:,44], htxt[:,45], htxt[:,46] ] ).T,   
            'p_com': np.array( [ htxt[:,50], htxt[:,51], htxt[:,52] ] ).T
          }

   # --- read header info and scale factor of outputs
   header = g.readsnap(simdir,0,0,header_only=1)
   hinv = 1. / header['hubble']
   if len(outputlist) > 0:        # assumes AHF output available for all snapshots
     print '   ... reading ' + simdir + '/' + outputlist
     file = open( simdir + '/' + outputlist )
     a = np.loadtxt(file)[::-1]   # from low to high redshift
     file.close()
     z = 1./a - 1.
     snapnum = np.arange(a.size)[::-1]
     j = np.argmin( np.abs(z-halo['z'][0]) )
     nsnap = halo['z'].size
     if np.min( np.abs(halo['z'][:] - z[j:j+nsnap]) ) > 1e-3:
       print 'problem with redshifts!!'
       return
     halo['z'][:] = z[j:j+nsnap]
     halo['snapnum'] = snapnum[j:j+nsnap]
     ascale = a[j:j+nsnap]
   else:
     print '\n\nWARNING !!! using redshifts from AHF file names!!!\n\n'
     ascale = 1./(1.+halo['z'])

   halo['Mvir'][:] *= hinv          # in Msun
   halo['Rvir'][:] *= hinv*ascale   # all in physical kpc
   for i in range(3):
      halo['pos'][:,i] *= hinv*ascale
      halo['p_mbp'][:,i] *= hinv*ascale*1e3 
      halo['p_com'][:,i] *= hinv*ascale
                                    # peculiar velocity is already in km/s

   if cap:   # don't allow Rvir to decrease with time

      Rvir_change = halo['Rvir'][0:-1] - halo['Rvir'][1:]
      ind = np.where( Rvir_change < 0 )[0]
      if ind.size > 0:
         #nw = 0
         while ind.size > 0:  
              halo['Mvir'][ind] = halo['Mvir'][ind+1]
              halo['Rvir'][ind] = halo['Rvir'][ind+1]
              Rvir_change = halo['Rvir'][0:-1] - halo['Rvir'][1:]
              ind = np.where( Rvir_change < 0 )[0]
              #print '...', nw
              #nw += 1

   if smooth > 0:  # smooth time evolution of halo properties

      halo['MvirOrig'] = halo['Mvir'].copy()
      halo['RvirOrig'] = halo['Rvir'].copy()
      halo['posOrig'] = halo['pos'].copy()
      halo['velOrig'] = halo['vel'].copy()
      #--- smooth coordinates and velocities
      #nbox = 11
      nbox = smooth
      for i in range(3):
         halo['pos'][:,i] = util.smooth( halo['pos'][:,i]/ascale, window_len=nbox, window='flat') * ascale
         halo['p_mbp'][:,i] = util.smooth( halo['p_mbp'][:,i]/ascale, window_len=nbox, window='flat') * ascale
         halo['p_com'][:,i] = util.smooth( halo['p_com'][:,i]/ascale, window_len=nbox, window='flat') * ascale
         halo['vel'][:,i] = util.smooth( halo['vel'][:,i], window_len=nbox, window='flat')


   print 'Done.\n'

   return halo



#############################################################################################
##############################################################################################

def read_skid( simdir, snapnum, FileType='grp', old=0, header={}, NewSkidDir='', snapdir='', snapname='snapshot', globstr='' ):

   time_start = time.time()

   if len(globstr) > 0:
     FilePath = glob( simdir + '/' + globstr + '.' + FileType )[0]
   else:
     if len(NewSkidDir) > 0:
        FilePath = simdir + '/'+ NewSkidDir + '/gal_' + g.snap_ext(snapnum) + '.' + FileType
     else:
        FilePath = simdir + '/skid/gal_' + g.snap_ext(snapnum) + '.' + FileType
 
   if not os.path.exists(FilePath):
      print '\nFile not found!! ' + FilePath
      return { 'redshift': -1 }

   print '\nReading ' + FilePath


   #**************************************************
   # *** ".grp": host galaxy ID for all particles ***
   #**************************************************

   if FileType == 'grp':

     df = pd.read_csv( FilePath, header=-1 )            # read from the first row (NpartTot)
 
     NpartTot = df.values[0,0]  
 
     data = { 'id':np.zeros(NpartTot,dtype=('int64')), 
              'GalID':df.values[1:,0] }

     # --- look for particle ids in the same order as Tipsy binary files
     start = 0
     print 'looking for particle IDs:'
     for Ptype in [0,1,4]:
        #P = g.readsnap(simdir,snapnum,Ptype,cosmological=1,skip_bh=1,header_only=0)
        P = read_ids_from_snap(simdir,snapnum,Ptype,snapshot_name=snapname)
        if P['k'] < 0:
           continue
        npart = P['id'].size
        data['id'][start:start+npart] = P['id']
        """
        if Ptype == 0:
           ind_test = np.where( data['GalID'][start:start+npart] > 0 )[0]
           if ind_test.size > 0:
              print '       ... Ptype =', Ptype, '  npart =', npart, '  (min,max)density =', \
              np.min(P['rho'][ind_test]) * UnitDensity_in_cgs / PROTONMASS, np.max(P['rho'][ind_test]) * UnitDensity_in_cgs / PROTONMASS
           else:
              print '       ... Ptype =', Ptype, '  npart =', npart
        else:
           print '       ... Ptype =', Ptype, '  npart =', npart
        """
        print '       ... Ptype =', Ptype, '  npart =', npart
        start += npart

     if (NpartTot != data['GalID'].size) or (NpartTot != start):
        print '  ---> wrong NpartTot = ', NpartTot, data['GalID'].size, start   

     time_end = time.time()
     print 'read_skid done in ... ', time_end - time_start, ' seconds'
     return data 


   #***********************************
   # *** ".gtp": galaxy properties ***
   #***********************************

   elif FileType == 'gtp':
     print 'in progress... not that useful actually'
     return None


   #***********************************************
   # *** ".stat": additional galaxy properties ***
   #***********************************************

#   1: group id, 2: number of particles, 3: total mass, 4: gas mass,
#   5: star mass, 6: maximum circular velocity, 7: 1/2 mass circular velocity
#   8: outer circular velocity, 9: radius of max circular velocity,
#   10: 1/2 mass radius, 11: outer radius, 12: 1-D velocity dispersion,
#   13-15: center, 16-18: CM velocity, 19-21: pos. of most bound particle.

   elif FileType == 'stat':
     
     f = open( FilePath )
     ftxt = np.loadtxt(f)
     f.close()

     if ftxt.size == 0:
        print '... empty file ...'
        return { 'redshift': -1 }

     if ftxt.ndim == 1:
        ftxt = np.reshape( ftxt, (1,ftxt.size) )

     # --- unit conversion factors in cgs
     if len(header) < 3:
       if len(snapdir) > 0:
          thisdir = snapdir
       else:
          thisdir = simdir
       header = g.readsnap(thisdir,snapnum,0,cosmological=1,skip_bh=1,header_only=1)
       if header['k'] == -1:
          snapname = glob( thisdir + '/snap*' + g.snap_ext(snapnum) + '*.hdf5')
          if len(snapname) > 0:
             snapname = snapname[0]
             snapname = snapname[ len(thisdir)+1 : -9 ]
             header = g.readsnap(thisdir,snapnum,0,cosmological=1,skip_bh=1,header_only=1,snapshot_name=snapname)

     ascale = 1. / ( 1. + header['redshift'] )
     h = header['hubble']
     hinv = 1./h

     unit_Length = 1e-3 * header['BoxSize'] * hinv * CM_PER_MPC                  
     unit_Time = np.sqrt( 8*np.pi/3) * CM_PER_MPC / ( 100. * h * CM_PER_KM )
     unit_Density = 1.8791e-29 * h**2                                            
     unit_Mass = unit_Density * unit_Length**3
     unit_Velocity = unit_Length / unit_Time

     if old == 0:
       data = { 'GalID': ftxt[:,0].astype(int),
                'redshift': header['redshift'],
                'npart': ftxt[:,1].astype(int),
                'ngas': ftxt[:,21].astype(int),
                'nstar': ftxt[:,22].astype(int),
                'Mass': ftxt[:,2] * unit_Mass / MSUN,                                            # in Msun
                'GasMass': ftxt[:,3] * unit_Mass / MSUN,
                'StarMass': ftxt[:,4] * unit_Mass / MSUN,
                'sfr': ftxt[:,24],                                                               # in Msun / yr
                'Rhalf': ftxt[:,9] * ascale * unit_Length / CM_PER_KPC,                          # in kpc (physical)
                'ReStar': ftxt[:,23] * ascale * unit_Length / CM_PER_KPC,
                'Rout': ftxt[:,10] * ascale * unit_Length / CM_PER_KPC,                          # in kpc (physical)
                'VcMax': ftxt[:,5] * ascale * unit_Velocity / CM_PER_KM,                         # in km/s (physical)  - from Stampede's version
                'VcRhalf': ftxt[:,6] * ascale * unit_Velocity / CM_PER_KM,
                'VcRout': ftxt[:,7] * ascale * unit_Velocity / CM_PER_KM,
                'p_den': ( np.array( [ ftxt[:,12], ftxt[:,13], ftxt[:,14] ] ).T + 0.5 ) * \
                         ascale * unit_Length / CM_PER_KPC,                                      # in kpc (physical)
                'p_phi': ( np.array( [ ftxt[:,18], ftxt[:,19], ftxt[:,20] ] ).T + 0.5 ) * \
                         ascale * unit_Length / CM_PER_KPC       ,                               # in kpc (physical)
                'v_CM': ( np.array( [ ftxt[:,15], ftxt[:,16], ftxt[:,17] ] ).T ) * \
                         ascale * unit_Velocity / CM_PER_KM }                                    # in km/s (physical)
     else:
       data = { 'GalID': ftxt[:,0].astype(int),
                'redshift': header['redshift'],
                'npart': ftxt[:,1].astype(int),
                'Mass': ftxt[:,2] * unit_Mass / MSUN,                                            # in Msun
                'GasMass': ftxt[:,3] * unit_Mass / MSUN,
                'StarMass': ftxt[:,4] * unit_Mass / MSUN,
                'Rhalf': ftxt[:,9] * ascale * unit_Length / CM_PER_KPC,                          # in kpc (physical)
                'Rout': ftxt[:,10] * ascale * unit_Length / CM_PER_KPC,                          # in kpc (physical)
                'p_den': ( np.array( [ ftxt[:,12], ftxt[:,13], ftxt[:,14] ] ).T + 0.5 ) * \
                         ascale * unit_Length / CM_PER_KPC,                                      # in kpc (physical)
                'p_phi': ( np.array( [ ftxt[:,18], ftxt[:,19], ftxt[:,20] ] ).T + 0.5 ) * \
                         ascale * unit_Length / CM_PER_KPC       ,                               # in kpc (physical)
                'v_CM': ( np.array( [ ftxt[:,15], ftxt[:,16], ftxt[:,17] ] ).T ) * \
                         ascale * unit_Velocity / CM_PER_KM }                                    # in km/s (physical)

     print 'Done.\n'
     return data


   else:
     print '  ---> what is going on?'
     return None


   print 'Done.\n'



##############################################################################################
##############################################################################################

def read_ids_from_snap(sdir,snum,ptype,
    snapshot_name='snapshot',
    extension='.hdf5',
    four_char=0):

    if (ptype<0): return {'k':-1};
    if (ptype>5): return {'k':-1};

    fname,fname_base,fname_ext = g.check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)
    if(fname=='NULL'): return {'k':-1}

    ## open file and parse its header information
    nL = 0 # initial particle point to start at
    if(fname_ext=='.hdf5'):
        file = h5py.File(fname,'r') # Open hdf5 snapshot file
        header_master = file["Header"] # Load header dictionary (to parse below)
        header_toparse = header_master.attrs
    else:
        print 'not an HDF5 file !!'
        return {'k':-1}

    npart = header_toparse["NumPart_ThisFile"]
    npartTotal = header_toparse["NumPart_Total"]
    numfiles = header_toparse["NumFilesPerSnapshot"]

    ids=np.zeros([npartTotal[ptype]],dtype=long)

    # loop over the snapshot parts to get the different data pieces
    for i_file in range(numfiles):
        if (numfiles>1):
            file.close()
            fname = fname_base+'.'+str(i_file)+fname_ext
            file = h5py.File(fname,'r') # Open hdf5 snapshot file
        input_struct = file
        npart = file["Header"].attrs["NumPart_ThisFile"]
        bname = "PartType"+str(ptype)+"/"

        # now do the actual reading
        if npart[ptype]>0:
          nR=nL + npart[ptype]
          ids[nL:nR]=input_struct[bname+"ParticleIDs"]
          nL = nR # sets it for the next iteration

        ## correct to same ID as original gas particle for new stars, if bit-flip applied
    if ((np.min(ids)<0) | (np.max(ids)>1.e9)):
        bad = (ids < 0) | (ids > 1.e9)
        ids[bad] += (1L << 31)

    file.close();

    return {'k':1, 'id':ids }




##############################################################################################
##############################################################################################


def read_blackhole( simdir, redo=0, old=0 ):


   outfile = simdir + '/blackhole_details/bhALL.hdf5'

   if old == 1:
      names = [ 'a', 'ID', 'Mass', 'BH_Mass', 'Mass_AlphaDisk', 'Mdot', 'Mdot_alphadisk', 'dt',
                'rho', 'e', 'Malt', 'Mgas', 'Mbulge', 'R0', 'p', 'v', 'j' ]
      formats = [ 'f8', 'i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                  'f8', 'f8', 'f8', 'f8', 'f8', 'f8', '(3,)f8', '(3,)f8', '(3,)f8' ]
   else:
      names = [ 'a', 'ID', 'Mass', 'BH_Mass', 'Mass_AlphaDisk', 'Mdot', 'Mdot_alphadisk', 'dt',
                'rho', 'e', 'Sfr', 'Mgas', 'Mstar', 'MgasBulge', 'MstarBulge', 'R0', 'p', 'v', 'Jgas', 'Jstar' ]
      formats = [ 'f8', 'i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                  'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', '(3,)f8', '(3,)f8', '(3,)f8', '(3,)f8' ]


   bh_template = np.zeros( 1, dtype={ 'names':names, 'formats':formats } )


   if (os.path.isfile(outfile)) and (redo == 0):


      print 'reading ' + outfile + ' ...'

      f = h5py.File(outfile, 'r')
      #print f.keys()
      bh = np.repeat( bh_template, f['a'].size )
      for keyname in f.keys():
          bh[keyname][:] = f[keyname][:]
      f.close()    

      #print 'Done.\n'

      return bh


   else:


      BHFiles = glob( simdir + '/blackhole_details/blackhole_details_*.txt')

      nfiles = len(BHFiles)

      if nfiles == 0:
        print 'WARNING! no blackhole_details files found...'
        return -1

      j = 0
      for FilePath in BHFiles:
         print j, ' Reading ' + FilePath, '...'      

         hfile = open( FilePath )
         #htxt = np.loadtxt( hfile )
         htxt = np.genfromtxt( hfile )
         hfile.close()

         if htxt.size == 0:
           continue

         if htxt.ndim == 1:
           nlines = 1
           htxt = np.reshape( htxt, (1,htxt.shape[0]) )
         else:
           nlines = htxt[:,0].size

         bh_tmp = np.repeat( bh_template, nlines )

         col = 0
         for this in names:

            if this in [ 'p', 'v', 'j', 'Jgas', 'Jstar' ]: 
              for k in xrange(3):
                 bh_tmp[this][:,k] = htxt[:,col]
                 col+=1
            else:
              bh_tmp[this][:] = htxt[:,col]
              col+=1

         if j == 0:
           bh = bh_tmp.copy()
         else:
           bh = np.concatenate((bh,bh_tmp))     

         j+=1


      # --- unit conversion ---

      header = g.readsnap(simdir,0,0,cosmological=1,skip_bh=1,header_only=1)
      h = header['hubble']

      ascale = bh['a'][:]

      bh['Mass'] *= ( UnitMass_in_Msun / h )            # in Msun
      bh['BH_Mass'] *= ( UnitMass_in_Msun / h )
      bh['Mass_AlphaDisk'] *= ( UnitMass_in_Msun / h )
      bh['Mgas'] *= ( UnitMass_in_Msun / h )
      if old == 1:
        bh['Malt'] *= ( UnitMass_in_Msun / h )
        bh['Mbulge'] *= ( UnitMass_in_Msun / h )
      else:
        bh['Mstar'] *= ( UnitMass_in_Msun / h )
        bh['MgasBulge'] *= ( UnitMass_in_Msun / h )
        bh['MstarBulge'] *= ( UnitMass_in_Msun / h )
   
      bh['Mdot'] *= UnitMdot_in_Msun_per_yr             # in Msun / yr
      bh['Mdot_alphadisk'] *= UnitMdot_in_Msun_per_yr

      bh['dt'] *= ( UnitTime_in_Myr / h )               # in Myr

      bh['R0'] /= h                                     # in kpc (physical already) 

      for k in range(3):
         bh['p'][:,k] *= ( ascale / h )                 # in kpc (physical)
         bh['v'][:,k] /= ascale                         # in Km/s (physical)
   

      # --- write data to HDF5 file ---

      f = h5py.File(outfile, 'w')
      for keyname in bh.dtype.names:
          f.create_dataset(keyname, data=bh[keyname])
      f.close()

      #print 'Done.\n'

      return bh


##############################################################################################
##############################################################################################


def read_blackhole_mergers( simdir, redo=0 ):


   outfile = simdir + '/blackhole_details/bhMRG.hdf5'


   names = [ 'a', 'ID_1', 'BH_Mass_1', 'p_1', 'ID_2', 'BH_Mass_2', 'p_2' ]

   formats = [ 'f8', 'i8', 'f8', '(3,)f8', 'i8', 'f8', '(3,)f8' ]

   bh_template = np.zeros( 1, dtype={ 'names':names, 'formats':formats } )


   if (os.path.isfile(outfile)) and (redo == 0):


      print 'reading ' + outfile + ' ...'

      f = h5py.File(outfile, 'r')
      bh = np.repeat( bh_template, f['a'].size )
      for keyname in f.keys():
          bh[keyname][:] = f[keyname][:]
      f.close()

      #print 'Done.\n'

      return bh


   else:


      BHFiles = glob( simdir + '/blackhole_details/bhmergers_*.txt')

      nfiles = len(BHFiles)

      if nfiles == 0:
        print 'WARNING! no bhmergers files found...'
        return -1

      j = 0
      for FilePath in BHFiles:
         print j, ' Reading ' + FilePath, '...'

         hfile = open( FilePath )
         htxt = np.loadtxt( hfile )
         hfile.close()

         if htxt.size == 0:
           continue

         if htxt.ndim == 1:
           nlines = 1
           htxt = np.reshape( htxt, (1,htxt.shape[0]) )
         else:
           nlines = htxt[:,0].size

         bh_tmp = np.repeat( bh_template, nlines )

         col = 0
         for this in names:

            if this in ['p_1', 'p_2']:
              for k in xrange(3):
                 bh_tmp[this][:,k] = htxt[:,col]
                 col+=1
            else:
              bh_tmp[this][:] = htxt[:,col]
              col+=1

         if j == 0:
           bh = bh_tmp.copy()
         else:
           bh = np.concatenate((bh,bh_tmp))

         j+=1


      # --- unit conversion ---

      if j > 0:

        header = g.readsnap(simdir,0,0,cosmological=1,skip_bh=1,header_only=1)
        h = header['hubble']

        bh['BH_Mass_1'] *= ( UnitMass_in_Msun / h )            # in Msun
        bh['BH_Mass_2'] *= ( UnitMass_in_Msun / h )

        # --- write data to HDF5 file ---

        f = h5py.File(outfile, 'w')
        for keyname in bh.dtype.names:
            f.create_dataset(keyname, data=bh[keyname])
        f.close()

        #print 'Done.\n'

        return bh

      else:

        return {}


##############################################################################################
##############################################################################################


def read_blackhole_winds( simdir, redo=0 ):


   outfile = simdir + '/blackhole_details/bhBAL.hdf5'


   names = [ 'a', 'ID', 'Mass', 'p', 'v', 'd', 'ID_bh', 'p_bh' ]

   formats = [ 'f8', 'i8', 'f8', '(3,)f8', '(3,)f8', '(3,)f8', 'i8', '(3,)f8' ]

   bh_template = np.zeros( 1, dtype={ 'names':names, 'formats':formats } )


   if (os.path.isfile(outfile)) and (redo == 0):


      print 'reading ' + outfile + ' ...'

      f = h5py.File(outfile, 'r')
      bh = np.repeat( bh_template, f['a'].size )
      for keyname in f.keys():
          bh[keyname][:] = f[keyname][:]
      f.close()

      #print 'Done.\n'

      return bh


   else:


      BHFiles = glob( simdir + '/blackhole_details/bhwinds_*.txt')

      nfiles = len(BHFiles)

      if nfiles == 0:
        print 'WARNING! no bhwinds files found...'
        return -1

      j = 0
      for FilePath in BHFiles:
         print j, ' Reading ' + FilePath, '...'

         hfile = open( FilePath )
         htxt = np.loadtxt( hfile )
         hfile.close()

         if htxt.size == 0:
           continue

         if htxt.ndim == 1:
           nlines = 1
           htxt = np.reshape( htxt, (1,htxt.shape[0]) )
         else:
           nlines = htxt[:,0].size

         bh_tmp = np.repeat( bh_template, nlines )

         col = 0
         for this in names:

            if this in ['p', 'v', 'd', 'p_bh']:
              for k in xrange(3):
                 bh_tmp[this][:,k] = htxt[:,col]
                 col+=1
            else:
              bh_tmp[this][:] = htxt[:,col]
              col+=1

         if j == 0:
           bh = bh_tmp.copy()
         else:
           bh = np.concatenate((bh,bh_tmp))

         j+=1


      # --- unit conversion ---

      if j > 0:

        header = g.readsnap(simdir,0,0,cosmological=1,skip_bh=1,header_only=1)
        h = header['hubble']

        bh['Mass'] *= ( UnitMass_in_Msun / h )            # in Msun

        # --- write data to HDF5 file ---

        f = h5py.File(outfile, 'w')
        for keyname in bh.dtype.names:
            f.create_dataset(keyname, data=bh[keyname])
        f.close()
        #print 'Done.\n'
        return bh

      else:

        return {}



##############################################################################################
##############################################################################################


def blackhole_track( id, bhALL, bhMRG=-1, bhBAL=-1 ):

   thebh = -1
   thebh_mrg = -1
   thebh_bal = -1

   indID = np.where( bhALL['ID'] == id )[0]
   if indID.size == 0:
      print 'black hole ID not found!!'
      return -1
   #print 'looking for black hole #', id
   thebh = bhALL[indID]
   #dummy, ind = np.unique( thebh['a'], return_index=True)
   #thebh = thebh[ind[::-1]]
   ind_sort = np.argsort( thebh['a'] )[::-1]
   thebh = thebh[ind_sort]                
   isfinite = np.isfinite( thebh['a'] )
   thebh = thebh[isfinite]

   if util.checklen(bhMRG) > 1:
     indID = np.where( bhMRG['ID_1'] == id )[0]
     if indID.size > 0:
       thebh_mrg = bhMRG[indID]
       dummy, ind = np.unique( thebh_mrg['a'], return_index=True)
       thebh_mrg = thebh_mrg[ind[::-1]]
     else:
       print 'black hole ID not found in bhMRG'

   if util.checklen(bhBAL) > 1:
     indID = np.where( bhBAL['ID_bh'] == id )[0]
     if indID.size > 0:
       thebh_bal = bhBAL[indID]
       dummy, ind = np.unique( thebh_bal['a'], return_index=True)
       thebh_bal = thebh_bal[ind[::-1]]
     else:
       print 'black hole ID not found in bhBAL'
   

   return thebh, thebh_mrg, thebh_bal



##############################################################################################
##############################################################################################


def integrate_eddington( bhstruct, Mbh_ini, eff, MstarMin=1e7, gasonly=False ):

   dt = 1e6 * bhstruct['dt'][:]  # in yr
   nstep = dt.size

   ind_mstar = np.where( bhstruct['ReStar']['MbulgeStar'][:] >= MstarMin )[0]
   istart = ind_mstar[-1]
   if istart > nstep-1:
     istart = nstep-1

   Mbh = np.zeros(nstep)
   Mbh[istart] = Mbh_ini

   for i in range(istart,0,-1):

      Mdot = 0.

      if gasonly:
        if bhstruct['100pc']['GasMass'][i] > 0:
          Mdot = eff * MdotEddington(Mbh[i])     
      else:
        Mdot = eff * MdotEddington(Mbh[i])

      Mbh[i-1] = Mbh[i] + Mdot * dt[i-1]

   return Mbh



def integrate_bondi_analytic( bhstruct, Mbh_ini, eff, EddLimit=10000., MstarMin=1e7, rho=0.1, cs=30. ):

   # rho = cm^-3
   # cs  = km/s

   dt = 1e6 * bhstruct['dt'][:]  # in yr
   nstep = dt.size

   # --- accretion rate per unit Mbh**(1/6)  [Msun/year]
   Mbh_dummy = 1.
   MdotBondiN = 4. * np.pi * G_UNIV**2 * (Mbh_dummy*MSUN)**2. * (rho*PROTONMASS) / (cs*CM_PER_KM )**3 * (SEC_PER_YEAR/MSUN)

   ind_mstar = np.where( bhstruct['ReStar']['MbulgeStar'][:] >= MstarMin )[0]
   istart = ind_mstar[-1]
   if istart > nstep-1:
     istart = nstep-1

   Mbh = np.zeros(nstep)
   Mbh[istart] = Mbh_ini

   for i in range(istart,0,-1):

      Mdot = eff * MdotBondiN * Mbh[i]**2.

      EddingtonLimit = MdotEddington(Mbh[i])
      if Mdot > EddLimit*EddingtonLimit:
        Mdot = EddLimit*EddingtonLimit

      Mbh[i-1] = Mbh[i] + Mdot * dt[i-1]

   return Mbh



def integrate_bondi( bhstruct, bondi, Mbh_ini, eff, EddLimit=1000, MstarMin=1e7, gasonly=False, saturate=False ):

   dt = 1e6 * bhstruct['dt'][:]  # in yr
   nstep = dt.size

   MdotBondiN = np.zeros(nstep+1)

   if gasonly:
     ind = np.where( (bondi['cs'] > 0) & (bhstruct['100pc']['GasMass'] > 0) )[0]
   else:
     ind = np.where( bondi['cs'] > 0 )[0]

   rho = bondi['rho'][ind]
   cs = bondi['cs'][ind]

   if saturate:
     rho_min = np.percentile( rho, 50 )
     ind_min = np.argmin( np.abs(rho-rho_min) )
     cs_min = cs[ind_min]
     ind_sat = np.where( rho < rho_min )[0]
     rho[ind_sat] = rho_min
     cs[ind_sat] = cs_min

   Mbh_dummy = 1.
   # --- accretion rate per unit Mbh**2  [Msun/year]
   G_UNIV_CodeUnits = G_UNIV / UnitLength_in_cm**3 * UnitMass_in_g * UnitTime_in_s**2
   MdotBondiN[ind] = 4. * np.pi * G_UNIV_CodeUnits**2 * (Mbh_dummy/UnitMass_in_Msun)**2. * rho / cs**(3./2.)
   MdotBondiN *= UnitMdot_in_Msun_per_yr

   ind_mstar = np.where( bhstruct['ReStar']['MbulgeStar'][:] >= MstarMin )[0]
   istart = ind_mstar[-1]
   if istart > nstep-1:
     istart = nstep-1

   Mbh = np.zeros(nstep)
   Mbh[istart] = Mbh_ini

   for i in range(istart,0,-1):

      Mdot = eff * MdotBondiN[i] * Mbh[i]**2.

      EddingtonLimit = MdotEddington(Mbh[i])
      if Mdot > EddLimit*EddingtonLimit:
        Mdot = EddLimit*EddingtonLimit

      Mbh[i-1] = Mbh[i] + Mdot * dt[i-1]

   return Mbh



def integrate_torque( bhstruct, Mbh_ini, eff, tag='r256', EddLimit=1000, MstarMin=1e7, precomputed=False, fd=-1, fgas=-1, Mgas=-1 ):

   dt = 1e6 * bhstruct['dt'][:]  # in yr
   nstep = dt.size

   if precomputed:

     MdotTorqueN = bhstruct[tag]['MdotTorqueN'][:]

   else:

     MdotTorqueN = np.zeros(nstep+1)
     if MdotTorqueN.size != bhstruct[tag]['MdotTorqueN'][:].size:
       print 'ahhhhhhhhhhhhhhhhrgggggggggg'
     R0 = bhstruct[tag]['R0'][:]
     GasMass = bhstruct[tag]['GasMass'][:]
     StarMass = bhstruct[tag]['StarMass'][:]
     MbulgeGas = bhstruct[tag]['MbulgeGas'][:]
     MbulgeStar = bhstruct[tag]['MbulgeStar'][:]
     Md = GasMass + StarMass - MbulgeStar                # all gas assigned to the disk!

     # --- check for test cases:
     if fd > 0:
       print ' ...TEST case fd > 0 !!'
       fgas = GasMass / Md
       Md = fd * (GasMass + StarMass)
       fd = np.repeat(fd, nstep+1)
     elif fgas > 0:
       print ' ...TEST case fgas > 0 !!'
       fd = Md / (GasMass + StarMass)
       fgas = np.repeat(fgas, nstep+1)
     elif Mgas > 0:
       print ' ...TEST case Mgas > 0 !!'
       GasMass = np.repeat(Mgas, nstep+1)
       Md = GasMass + StarMass - MbulgeStar
       fgas = GasMass / Md
       fd = Md / (GasMass + StarMass)
     else:
       fgas = GasMass / Md
       fd = Md / (GasMass + StarMass)

     ind = np.where( (GasMass > 0) & (Md > 0) & (R0 > 0) )[0]
     f0 = 0.31 * fd[ind]**2. * (Md[ind]/1e9)**(-1./3.)
     alphaK = 5.
     Mbh_dummy = 1.
     # --- accretion rate per unit Mbh**(1/6)  [Msun/year]
     MdotTorqueN[ind] = alphaK * fd[ind]**(5./2.) * (Mbh_dummy/1e8)**(1./6.) * (Md[ind]/1e9) * (R0[ind]*1000./1e2)**(-3./2.) * (1 + f0/fgas[ind])**(-1)

   #ind_mstar = np.where( bhstruct['0.1Rvir']['StarMass'][:] > MstarMin )[0]
   #istart = ind_mstar[-1]
   #if istart > nstep-1:
   #  istart = nstep-1

   ind_mstar = np.where( bhstruct['ReStar']['MbulgeStar'][:] >= MstarMin )[0]
   istart = ind_mstar[-1]
   if istart > nstep-1:
     istart = nstep-1

   Mbh = np.zeros(nstep)
   Mbh[istart] = Mbh_ini

   for i in range(istart,0,-1):

      MdotTorque = eff * MdotTorqueN[i] * Mbh[i]**(1./6.)

      EddingtonLimit = MdotEddington(Mbh[i])
      if MdotTorque > EddLimit*EddingtonLimit:
        MdotTorque = EddLimit*EddingtonLimit

      Mbh[i-1] = Mbh[i] + MdotTorque * dt[i-1] 

   return Mbh



def integrate_torque_variables( bhstruct, Mbh_ini, eff, tag='r256', EddLimit=1000, MstarMin=1e7 ):

   dt = 1e6 * bhstruct['dt'][:]  # in yr
   nstep = dt.size

   MdotTorqueN = np.zeros(nstep+1)
   f0 = np.zeros(nstep+1)

   R0 = bhstruct[tag]['R0'][:]
   GasMass = bhstruct[tag]['GasMass'][:]
   StarMass = bhstruct[tag]['StarMass'][:]
   MbulgeGas = bhstruct[tag]['MbulgeGas'][:]
   MbulgeStar = bhstruct[tag]['MbulgeStar'][:]
   Md = GasMass + StarMass - MbulgeStar                # all gas assigned to the disk!
   fgas = GasMass / Md
   fd = Md / (GasMass + StarMass)
   ind = np.where( (GasMass > 0) & (Md > 0) & (R0 > 0) )[0]
   f0[ind] = 0.31 * fd[ind]**2. * (Md[ind]/1e9)**(-1./3.)
   alphaK = 5.
   Mbh_dummy = 1.
   # --- accretion rate per unit Mbh**(1/6)  [Msun/year]
   MdotTorqueN[ind] = alphaK * fd[ind]**(5./2.) * (Mbh_dummy/1e8)**(1./6.) * (Md[ind]/1e9) * (R0[ind]*1000./1e2)**(-3./2.) * (1 + f0[ind]/fgas[ind])**(-1)

   ind_mstar = np.where( bhstruct['0.1Rvir']['StarMass'][:] > MstarMin )[0]
   istart = ind_mstar[-1]
   if istart > nstep-1:
     istart = nstep-1

   Mbh = np.zeros(nstep)
   Mbh[istart] = Mbh_ini

   Mdot = np.zeros(nstep)

   for i in range(istart,0,-1):

      MdotTorque = eff * MdotTorqueN[i] * Mbh[i]**(1./6.)

      EddingtonLimit = MdotEddington(Mbh[i])
      if MdotTorque > EddLimit*EddingtonLimit:
        MdotTorque = EddLimit*EddingtonLimit

      Mdot[i-1] = MdotTorque
      Mbh[i-1] = Mbh[i] + MdotTorque * dt[i-1]

   return { 'Mbh':Mbh, 'Mdot':Mdot, 'R0':R0, 'Md':Md, 'fd':fd, 'fgas':fgas, 'f0':f0, 'GasMass':GasMass, 'dt':dt }



def integrate_mdot( bhstruct, Mbh_ini, eff, tag='r256', mode='SFR', EddLimit=1000, MstarMin=1e7 ):

   dt = 1e6 * bhstruct['dt'][:]  # in yr
   nstep = dt.size

   if mode == 'SFR':
     SFR = bhstruct[tag]['SFR'][:]
   elif mode in ['FreeFall','DynamicalTime']:
     R0 = bhstruct[tag]['R0'][:]
     GasMass = bhstruct[tag]['GasMass'][:]
     StarMass = bhstruct[tag]['StarMass'][:]
     mbh = bhstruct['bh']['mbh'][:]
     Mtot = GasMass + StarMass + mbh
     ind = np.where( (GasMass > 0) & (R0>0) )[0]
     Tff = np.zeros(nstep+1)
     if mode == 'FreeFall':
       rho = ( GasMass[ind]*MSUN ) / ( (4./3.) * np.pi * (R0[ind]*CM_PER_KPC)**3 )  # cgs
       Tff[ind] = np.sqrt( 3.*np.pi / (32.*G_UNIV*rho) ) / SEC_PER_YEAR        # years
     else:
       #Tff[ind] = np.sqrt( (R0[ind]*CM_PER_KPC)**3 / (2.*G_UNIV*Mtot[ind]*MSUN) ) / SEC_PER_YEAR      
       Tff[ind] = np.sqrt( (R0[ind]*CM_PER_KPC)**3 / (G_UNIV*Mtot[ind]*MSUN) ) / SEC_PER_YEAR

   #ind_mstar = np.where( bhstruct['0.1Rvir']['StarMass'][:] > MstarMin )[0]
   #istart = ind_mstar[-1]
   ind_mstar = np.where( bhstruct['ReStar']['MbulgeStar'][:] >= MstarMin )[0]
   istart = ind_mstar[-1] 
   if istart > nstep-1:
     istart = nstep-1

   Mbh = np.zeros(nstep)
   Mbh[istart] = Mbh_ini

   for i in range(istart,0,-1):

      Mdot = 0

      if mode == 'SFR':
        Mdot = eff * SFR[i]
      elif mode in ['FreeFall','DynamicalTime']:
        if Tff[i] > 0:
          Mdot = eff * GasMass[i] / Tff[i]

      EddingtonLimit = MdotEddington(Mbh[i])
      if Mdot > EddLimit*EddingtonLimit:
        Mdot = EddLimit*EddingtonLimit

      Mbh[i-1] = Mbh[i] + Mdot * dt[i-1]

   return Mbh


def integrate_sfr( gstruct, M_ini, tag='r256', MstarMin=1e7 ):

   dt = 1e6 * gstruct['dt'][:]  # in yr
   nstep = dt.size

   SFR = gstruct[tag]['SFR'][:]
   ind_mstar = np.where( gstruct['0.1Rvir']['StarMass'][:] > MstarMin )[0]
   istart = ind_mstar[-1]
   if istart > nstep-1:
     istart = nstep-1

   Mass = np.zeros(nstep)
   Mass[istart] = M_ini

   for i in range(istart,0,-1):
      Mass[i-1] = Mass[i] + SFR[i] * dt[i-1]

   return Mass


##############################################################################################
##############################################################################################

# HI mass fraction in post-processing (from Claude-Andre):

# Input: T: temperature (K)
#
# Output: Case A HII recombination coefficient (cm^3 s^-1) used by
#         Gadget (see Katz et al. 1996).
def HII_recomb_case_A_Gadget(T):
  result = 8.4*10.0**-11.0
  result = result/np.sqrt(T)
  result = result*(T/1000.0)**-0.2
  result = result/(1.0 + (T/(10.0**6.0))**0.7)
  return result

# Equation (1) in Rahmati & Schaye (2013) for self-shielding correction.
def Rahmati_Gamma_factor(nH, Gamma_bkg_12, sigma_nuHI_bar):
  nHss = 6.73e-3*(sigma_nuHI_bar/(2.49e-18))**-0.6667*(Gamma_bkg_12)**0.6667
  fact_ = 0.98*(1.0 + (nH/nHss)**1.64)**-2.28 + 0.02*(1.0 + nH/nHss)**-0.84
  return fact_


def HI_mass_fraction(P):
  # Frequency-averaged HI photoionization cross section (cm^-2). Default value
  # in Rahmati+15 paper.
  sigma_nuHI_bar = 2.49e-18

  # UVB HI photoionization rate in units of 10^-12 s^-1. Valid at z=2, from FG+09.
  Gamma_bkg_12 = 0.6

  # In what follows, nHII_cgs, ne_cgs are the HII and total electron number densities
  # (taking into a helium) in units cm^-3 that you compute from the particles. The GIZMO
  # snapshots should contain an "electron fraction" called 'ne.' The total electron number
  # density is nH*ne, where nH is the total H number density. ne will have typical values
  # ~1.2 (greater than 1) because of the electrons from helium.

  T = g.gas_temperature(P['u'],P['ne'])

  nH = 0.75 * P['rho'][:] * UnitDensity_in_cgs / PROTONMASS      # cm^(-3) 

  ne_cgs = nH * P['ne'][:]

  nHII_cgs = nH

  # HI mass fraction in the optically thin, nearly fully ionized limit (valid for low-column systems):
  nhi_thin = HII_recomb_case_A_Gadget(T)*nHII_cgs*ne_cgs/(Gamma_bkg_12*10.0**-12.0)

  # Compute first-order self-shilidng correction factor.
  Rahmati_fact = Rahmati_Gamma_factor(nH, Gamma_bkg_12, sigma_nuHI_bar)

  # HI mass fraction to use (of total H), approximating the effects of self-shielding.
  nhi = np.minimum(nhi_thin/Rahmati_fact, 1.0)

  return nhi


##############################################################################################
##############################################################################################

# adapted from Phil's repo

def calculate_zoom_center(sdir,snum,cen=[0.,0.,0.],clip_size=2.e10):
    rgrid=np.array([1.0e10,1000.,700.,500.,300.,200.,100.,70.,50.,30.,20.,10.,5.,2.5,1.,0.5]);
    rgrid=rgrid[rgrid <= clip_size];
    Ps=g.readsnap(sdir,snum,4,cosmological=1);
    n_new=Ps['m'].shape[0];
    if (n_new > 1):
        pos=Ps['p']; x0s=pos[:,0]; y0s=pos[:,1]; z0s=pos[:,2];
    Pg=g.readsnap(sdir,snum,0,cosmological=1);
    rho=np.array(Pg['rho'])*407.5;
    if (rho.shape[0] > 0):
        pos=Pg['p']; x0g=pos[:,0]; y0g=pos[:,1]; z0g=pos[:,2];
    rho_cut=1.0e-5;
    cen=np.array(cen);

    for i_rcut in range(len(rgrid)):
        for j_looper in range(5):
            if (n_new > 1000):
                x=x0s; y=y0s; z=z0s;
            else:
                ok=(rho > rho_cut);
                x=x0g[ok]; y=y0g[ok]; z=z0g[ok];
            x=x-cen[0]; y=y-cen[1]; z=z-cen[2];
            r = np.sqrt(x*x + y*y + z*z);
            ok = (r < rgrid[i_rcut]);
            if (len(r[ok]) > 1000):
                x=x[ok]; y=y[ok]; z=z[ok];
                if (i_rcut <= len(rgrid)-5):
                    cen+=np.array([np.median(x),np.median(y),np.median(z)]);
                else:
                    cen+=np.array([np.mean(x),np.mean(y),np.mean(z)]);
            else:
                if (len(r[ok]) > 200):
                    x=x[ok]; y=y[ok]; z=z[ok];
                    cen+=np.array([np.mean(x),np.mean(y),np.mean(z)]);

    return cen;



##############################################################################################
##############################################################################################

# adapted from Phil's repo

def fcor(x):
    return np.array(x,dtype='f',ndmin=1)
def vfloat(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float));
def cfloat(x):
    return ctypes.c_float(x);
def checklen(x):
    return len(np.array(x,ndmin=1));
def ok_scan(input,xmax=1.0e10,pos=0):
    if (pos==1):
        return (np.isnan(input)==False) & (abs(input)<=xmax) & (input > 0.);
    else:
        return (np.isnan(input)==False) & (abs(input)<=xmax);


def get_particle_hsml( x, y, z, DesNgb=32, Hmax=0.):
    x=fcor(x); y=fcor(y); z=fcor(z); N=checklen(x);
    ok=(ok_scan(x) & ok_scan(y) & ok_scan(z)); x=x[ok]; y=y[ok]; z=z[ok];
    if(Hmax==0.):
        dx=np.max(x)-np.min(x); dy=np.max(y)-np.min(y); dz=np.max(z)-np.min(z); ddx=np.max([dx,dy,dz]);
        Hmax=5.*ddx*(np.float(N)**(-1./3.)); ## mean inter-particle spacing

    ## load the routine we need
    exec_call=util.return_python_routines_cdir()+'/StellarHsml/starhsml.so'
    h_routine=ctypes.cdll[exec_call];

    h_out_cast=ctypes.c_float*N; H_OUT=h_out_cast();
    ## main call to the hsml-finding routine
    h_routine.stellarhsml( ctypes.c_int(N), \
        vfloat(x), vfloat(y), vfloat(z), ctypes.c_int(DesNgb), \
        ctypes.c_float(Hmax), ctypes.byref(H_OUT) )

    ## now put the output arrays into a useful format
    h = np.ctypeslib.as_array(np.copy(H_OUT));
    return h;


def make_2Dgrid( x, y, xrange,
                 yrange=0, weight1=0, weight2=0, weight3=0, hsml=0,
                 pixels=128 ):

    xrange=np.array(xrange);
    if (checklen(yrange) <= 1): yrange=1.0*xrange;
    xmin=xrange[0]; xmax=xrange[1]; ymin=yrange[0]; ymax=yrange[1];
    xlen=0.5*(xmax-xmin); ylen=0.5*(ymax-ymin)

    if (checklen(weight1) <= 1): weight1=0.*x+1.; weight1 /= math.fsum(weight1);
    if (checklen(hsml) <= 1): hsml=0.*x+xlen/100.
    aspect_ratio = ylen/xlen;

    ## set some basic initial values
    xypixels=pixels; xpixels=xypixels; ypixels=np.around(float(xpixels)*aspect_ratio).astype(int);

    ## load the routine we need
    exec_call=util.return_python_routines_cdir()+'/SmoothedProjPFH/allnsmooth.so'
    smooth_routine=ctypes.cdll[exec_call];
    ## make sure values to be passed are in the right format
    N=checklen(x); x=fcor(x); y=fcor(y); M1=fcor(weight1); M2=fcor(weight2); M3=fcor(weight3); H=fcor(hsml)
    xpixels=np.int(xpixels); ypixels=np.int(ypixels)
    ## check for whether the optional extra weights are set
    NM=1;
    if( (checklen(M2)==checklen(M1)) & (checklen(M1)>1) ):
        NM=2;
        if(checklen(M3)==checklen(M1)):
            NM=3;
        else:
            M3=np.copy(M1);
    else:
        M2=np.copy(M1);
        M3=np.copy(M1);
    ## initialize the output vector to recieve the results
    XYpix=xpixels*ypixels; MAP=ctypes.c_float*XYpix; MAP1=MAP(); MAP2=MAP(); MAP3=MAP();
    ## main call to the imaging routine
    smooth_routine.project_and_smooth( \
        ctypes.c_int(N), \
        vfloat(x), vfloat(y), vfloat(H), \
        ctypes.c_int(NM), \
        vfloat(M1), vfloat(M2), vfloat(M3), \
        cfloat(xmin), cfloat(xmax), cfloat(ymin), cfloat(ymax), \
        ctypes.c_int(xpixels), ctypes.c_int(ypixels), \
        ctypes.byref(MAP1), ctypes.byref(MAP2), ctypes.byref(MAP3) );
    ## now put the output arrays into a useful format
    MassMap1=np.ctypeslib.as_array(MAP1).reshape([xpixels,ypixels]);
    MassMap2=np.ctypeslib.as_array(MAP2).reshape([xpixels,ypixels]);
    MassMap3=np.ctypeslib.as_array(MAP3).reshape([xpixels,ypixels]);

    if (NM==1):
      return MassMap1
    elif (NM==2):
      return MassMap1,MassMap2
    else:
      return MassMap1,MassMap2,MassMap3


def clip_2Dgrid( map1, map2=0 ):

    if map1.max() <= 0:
      if ( checklen(map1)==checklen(map2) ):
        return map1, map2
      else:
        return map1

    ind_bad = (map1 <= 0.)
    map1_min = np.min(map1[map1 > 0.])
    map1[ind_bad] = map1_min

    if ( checklen(map1)==checklen(map2) ):
       new_map = map2 / map1
       map2_min=np.min(map2[map2 > 0.]);
       new_map[ind_bad] = (map2[ind_bad] + map2_min) / map1_min;
       return map1, new_map
    else:
       return map1


# from: http://scipy-cookbook.readthedocs.io/items/Rebinning.html
def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported."
        return None



##############################################################################################
##############################################################################################

# from http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py

def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def write_hdf5_from_dict( var, outfile ):
   if os.path.isfile(outfile):
     os.remove(outfile)
   file = h5py.File(outfile, 'w')
   for grname in var.keys():
      if type(var[grname]) is dict:
        group = file.create_group( grname )
        for keyname in var[grname].keys():
           group.create_dataset(keyname, data=var[grname][keyname])
      else:
        file.create_dataset(grname, data=var[grname])
   file.close()
   return



##############################################################################################
##############################################################################################


def MdotEddington(Mbh):

   radeff = 0.1       # radiative efficiency

   MdotEdd = 4. * np.pi * G_UNIV * Mbh * PROTONMASS / ( radeff * SIGMA_T * SPEED_OF_LIGHT  ) * SEC_PER_YEAR  # in Msun / year   (if Mbh in Msun)

   return MdotEdd



##############################################################################################
##############################################################################################


def Haring04(Mbulge):

   return 10.**( 8.20 ) * ( Mbulge/1e11 )**(1.12)


def Kormendy2013(Mbulge):

   return 0.49 * 1e9 * ( Mbulge/1e11 )**(1.16)


def McConnell2013(Mbulge):

   return 10.**( 8.46 ) * ( Mbulge/1e11 )**(1.05)


def Reines2015(Mstar, sample='inactive'):

   if sample == 'agn':
     return 10.**( 7.45 ) * ( Mstar/1e11 )**(1.05)
   else:
     return 10.**( 8.95 ) * ( Mstar/1e11 )**(1.40)


def Savorgnan2016( Mstar, sample='all'):

   Mbh_early = 10.**( 8.56 + 1.04*np.log10(Mstar/10.**10.81) )
   Mbh_late = 10.**( 7.24 + 2.28*np.log10(Mstar/10.**10.05) )

   ind = np.where( Mbh_late - Mbh_early >= 0 )[0]
   Mbh_all =  np.append( Mbh_late[:ind[0]], Mbh_early[ind[0]:] )

   if Mbh_all.size != Mstar.size:
     print 'ahhhh.... check Savorgnan2016 !!!'

   if sample == 'early':
     return Mbh_early
   elif sample == 'late':
     return Mbh_late
   elif sample == 'all':
     return Mbh_all



##############################################################################################
##############################################################################################


def Chen2013_MdotToSFR( logMdot ):
  
   logSFR = ( logMdot + 3.72 ) / 1.05

   return logSFR


def Netzer2009( logMdot ):

  epsi = 0.1
  Lsun = 3.9e33     # in erg/s

  logSFR = np.log10( 1.09e-10 * 1e41 / Lsun ) + 0.8 * np.log10( epsi * SPEED_OF_LIGHT**2 * (MSUN/SEC_PER_YEAR) / 1e41 ) + 0.8 * logMdot

  return logSFR



##############################################################################################
##############################################################################################


def MainSequence_Speagle2014( logMstar, t ):


   logSFR = (0.84 - 0.026*t) * logMstar  -  (6.51 - 0.11*t)

   return logSFR



##############################################################################################
##############################################################################################


def MstarMhalo_Moster2013( Mhalo, z ):

   a = 1. / ( 1. + z )

   M10 = 11.590 
   M11 = 1.195 
   N10 = 0.0351 
   N11 = -0.0247 
   beta10 = 1.376 
   beta11 = -0.826 
   gama10 = 0.608 
   gama11 = 0.329   

   M10_err = 0.236 
   M11_err = 0.353 
   N10_err = 0.0058 
   N11_err = 0.0069 
   beta10_err = 0.153 
   beta11_err = 0.225 
   gama10_err = 0.059 
   gama11_err = 0.173

   M1 = 10.**( M10 + M11*(1.-a) )
   N = N10 + N11*(1.-a)
   beta = beta10 + beta11*(1.-a)
   gama = gama10 + gama11*(1.-a)

   Mstar = Mhalo * 2. * N * ( (Mhalo/M1)**(-beta) + (Mhalo/M1)**(gama) )**(-1.)

   return Mstar



##############################################################################################
##############################################################################################


def SFRD_Hopkins06(z):

   h = 0.7
   a = 0.017
   b = 0.13
   c = 3.3
   d = 5.3

   return (a + b*z) * h / ( 1.0 + (z/c)**d )


def SFRD_Madau2014(z):

   return 0.015 * (1. + z)**2.7 / ( 1. + ( (1. + z )/2.9 )**5.6 )



##############################################################################################
##############################################################################################


def SMF_Baldry2012( StarMass ):

   if checklen(StarMass) > 0:
     s = { 'Mstar':10.**(10.66), 'PhiStar1':3.96e-3, 'alpha1':-0.35, 'PhiStar2':0.79e-3, 'alpha2':-1.47 }   
     SMF = StarMass * np.exp(-1.*StarMass/s['Mstar']) * (  s['PhiStar1']*(StarMass/s['Mstar'])**s['alpha1'] +
                                                           s['PhiStar2']*(StarMass/s['Mstar'])**s['alpha2'] )  / ( s['Mstar'] * np.log10(np.exp(1)) )
     return SMF
   else:
     dfile = open( 'Baldry2012.txt' )
     dtxt = np.loadtxt(dfile)
     dfile.close()
     logMstar = dtxt[:,0]
     Phi = dtxt[:,2] * 1e-3
     Phi_err = dtxt[:,3] * 1e-3
     return logMstar, Phi, Phi_err


def SMF_Muzzin2013( LogStarMass, redshift ):

   if (redshift>2) and (redshift<2.5):

      s = { 'LogStarMassMin':10.54, 'LogMstar':11.00, 'LogMstarErr':0.02, 'PhiStar':2.94, 'PhiStarErr':0.40, 'Alpha':-1.2 }

      SMF = 1e-4 * np.log(10.) * s['PhiStar'] * 10.**( (LogStarMass - s['LogMstar'])*(1 + s['Alpha']) ) * np.exp( -10.**(LogStarMass - s['LogMstar']) )

      return SMF, s 

   else:
      print 'wrong redshift range!'
      return -1


def SMF_Tomczak2014( LogStarMass, redshift ):

   """
   Best-fit Double-Schechter Parameters to Tomczaket al. (2014) Stellar Mass Functions (Table 2)
   a In units of Msun.
   b In units of Mpc-3 dex-1.
   Total (Quiescent + Star-forming)	
   Redshift	Log(M*)^a	alpha_1	Log()^b	alpha_2	Log()^b		
   0.20 < z < 0.50	10.78 +or- 0.11	-0.98 +or- 0.24	-2.54 +or- 0.12	-1.90 +or- 0.36	-4.29 +or- 0.55	0.3	
   0.50 < z < 0.75	10.70 +or- 0.10	-0.39 +or- 0.50	-2.55 +or- 0.09	-1.53 +or- 0.12	-3.15 +or- 0.23	0.5	
   0.75 < z < 1.00	10.66 +or- 0.13	-0.37 +or- 0.49	-2.56 +or- 0.09	-1.61 +or- 0.16	-3.39 +or- 0.28	0.6	
   1.00 < z < 1.25	10.54 +or- 0.12	0.30 +or- 0.65	-2.72 +or- 0.10	-1.45 +or- 0.12	-3.17 +or- 0.19	0.8	
   1.25 < z < 1.50	10.61 +or- 0.08	-0.12 +or- 0.49	-2.78 +or- 0.08	-1.56 +or- 0.16	-3.43 +or- 0.23	0.3	
   1.50 < z < 2.00	10.74 +or- 0.09	0.04 +or- 0.62	-3.05 +or- 0.11	-1.49 +or- 0.14	-3.38 +or- 0.20	0.8	
   2.00 < z < 2.50	10.69 +or- 0.29	1.03 +or- 1.64	-3.80 +or- 0.30	-1.33 +or- 0.18	-3.26 +or- 0.23	0.4	
   2.50 < z < 3.00	10.74 +or- 0.31	1.62 +or- 1.88	-4.54 +or- 0.41	-1.57 +or- 0.20	-3.69 +or- 0.28	1.3	
   """

   s = { 'RedshiftIntervals':np.array( [  0.20,  0.50,  0.75,  1.00,  1.25,  1.50,  2.00,  2.50, 3.00 ] ),
                  'LogMstar':np.array( [ 10.78, 10.70, 10.66, 10.54, 10.61, 10.74, 10.69, 10.74 ] ), 
               'LogMstarErr':np.array( [  0.11,  0.10,  0.13,  0.12,  0.08,  0.09,  0.29,  0.31 ] ),
                   'Alpha_1':np.array( [ -0.98, -0.39, -0.37,  0.30, -0.12,  0.04,  1.03,  1.62 ] ), 
                'AlphaErr_1':np.array( [  0.24,  0.50,  0.49,  0.65,  0.49,  0.62,  1.64,  1.88 ] ), 
              'LogPhiStar_1':np.array( [ -2.54, -2.55, -2.56, -2.72, -2.78, -3.05, -3.80, -4.54 ] ), 
           'LogPhiStarErr_1':np.array( [  0.12,  0.09,  0.09,  0.10,  0.08,  0.11,  0.30,  0.41 ] ),
                   'Alpha_2':np.array( [ -1.90, -1.53, -1.61, -1.45, -1.56, -1.49, -1.33, -1.57 ] ), 
                'AlphaErr_2':np.array( [  0.36,  0.12,  0.16,  0.12,  0.16,  0.14,  0.18,  0.20 ] ), 
              'LogPhiStar_2':np.array( [ -4.29, -3.15, -3.39, -3.17, -3.43, -3.38, -3.26, -3.69 ] ), 
           'LogPhiStarErr_2':np.array( [  0.55,  0.23,  0.28,  0.19,  0.23,  0.20,  0.23,  0.28 ] ),
            'LogStarMassMin':np.array( [  8.00,  8.25,  8.50,  8.75,  8.75,  9.00,  9.25,  9.50 ] ) }


   ind = np.where( (s['RedshiftIntervals'][:-1] <= redshift) & (s['RedshiftIntervals'][1:] >= redshift) )[0]
   ind = ind[0]

   if ind >= 0:
      """
      s = { 'LogStarMassMin':9, 'LogMstar':10.69, 'LogMstarErr':0.29, 
            'Alpha_1':1.03, 'AlphaErr_1':1.64, 'LogPhiStar_1':-3.80, 'LogPhiStarErr_1':0.3, 
            'Alpha_2':-1.33, 'AlphaErr_2':0.18, 'LogPhiStar_2':-3.26, 'LogPhiStarErr_2':0.23 }
      SMF = np.log(10.) * np.exp( -10.**(LogStarMass - s['LogMstar']) ) * 10.**(LogStarMass - s['LogMstar']) * ( 
               10**s['LogPhiStar_1']*10.**( (LogStarMass-s['LogMstar'])*s['Alpha_1'] )  +  
               10**s['LogPhiStar_2']*10.**( (LogStarMass-s['LogMstar'])*s['Alpha_2'] )  )
      """

      SMF = np.log(10.) * np.exp( -10.**(LogStarMass - s['LogMstar'][ind]) ) * 10.**(LogStarMass - s['LogMstar'][ind]) * (
               10**s['LogPhiStar_1'][ind]*10.**( (LogStarMass-s['LogMstar'][ind])*s['Alpha_1'][ind] )  +
               10**s['LogPhiStar_2'][ind]*10.**( (LogStarMass-s['LogMstar'][ind])*s['Alpha_2'][ind] )  )

      return { 'SMF':SMF, 'RedshiftInterval':[ s['RedshiftIntervals'][ind], s['RedshiftIntervals'][ind+1] ], 'LogStarMassMin':s['LogStarMassMin'][ind] }

   else:
      print 'wrong redshift range!'
      return -1






##############################################################################################
##############################################################################################



def hubble_z(redshift, h=0.702, Omega0=0.272, OmegaLambda=0.728):

   # return Hubble factor in 1/sec for a given redshift

   ascale = 1. / ( 1. + redshift )
   hubble_a = HUBBLE * h * np.sqrt( Omega0 / ascale**3 + (1. - Omega0 - OmegaLambda) / ascale**2 + OmegaLambda )     # in 1/sec !!

   return hubble_a


def rho_crit_z(redshift):

   return 3 * hubble_z(redshift)**2 / ( 8 * np.pi * G )           # in cgs




##############################################################################################
##############################################################################################

# from http://matplotlib.1069221.n5.nabble.com/scatter-plot-individual-alpha-values-td21106.html

class LinearColormap(LinearSegmentedColormap):

    def __init__(self, name, segmented_data, index=None, **kwargs):
        if index is None:
            # If index not given, RGB colors are evenly-spaced in colormap.
            index = np.linspace(0, 1, len(segmented_data['red']))
        for key, value in segmented_data.iteritems():
            # Combine color index with color values.
            segmented_data[key] = zip(index, value)
        segmented_data = dict((key, [(x, y, y) for x, y in value])
                              for key, value in segmented_data.iteritems())
        LinearSegmentedColormap.__init__(self, name, segmented_data, **kwargs)


def colorbar_index(ncolors, labels, cmap, pad=0, title=''):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, pad=pad)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(labels)
    colorbar.set_label(title)

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in xrange(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)



def my_legend(leg,width=0, framewidth=0, colors=[]):
    """Color legend texts based on color of corresponding lines"""
    
    if len(leg.get_lines()) > 0:
      for line, txt in zip(leg.get_lines(), leg.get_texts()):
         txt.set_color(line.get_color())

    if len(colors) > 0:
      k = 0
      for txt in leg.get_texts():
         txt.set_color(colors[k])
         k += 1

    if width > 0:
      for l in leg.legendHandles:            
          l.set_linewidth(width)

    if framewidth > 0:
      leg.get_frame().set_linewidth(framewidth)



# from http://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc




##############################################################################################
##############################################################################################


def time_average( redshift, variable, time ):

   nbox = 100

   t = 1e3 * util.age_of_universe( redshift, h=h, Omega_M=Omega0 )   # in Myr
   dt = float(time) / nbox 
   t_new = np.linspace( t.min(), t.max(), (t.max()-t.min())/dt )

   var_new = np.interp( t_new, t[::-1], variable[::-1] )
   var_new = util.smooth( var_new, window_len=nbox, window='flat' )

   variable_new = np.interp( t, t_new, var_new )

   return variable_new




def gethist( values, bins=10, density=False, cumulative=False, box=True ):

   y, x = np.histogram( values, bins=bins, density=density )

   if cumulative:
      y = np.cumsum( y )

   if box:                                    # plot box-style histogram as line
      newy = np.repeat( y, 2 )
      newy = np.insert( newy, 0, 0. )
      newy = np.append( newy, 0. )
      newx = np.repeat(x,2)
   else:                                      # y, x values of "histogram centers"
      newy = y.astype(float)
      newx = ( x[1:] + x[:-1] ) / 2.          
   
   return newy, newx



def medianbin( x, y, bins, value=[25,50,75] ):

   if util.checklen(bins) == 1:
     Nbins = bins
     Dbin = ( np.max(x) - np.min(x) ) / float(Nbins)
     XbinS = np.min(x) + Dbin * np.arange(Nbins+1)
     xbin = XbinS[0:Nbins] + 0.5 * Dbin
   else:
     Nbins = bins.size - 1
     XbinS = bins
     xbin = ( XbinS[:-1] + XbinS[1:] ) / 2.

   nval = len(value)
   ymed = np.zeros([nval,Nbins])

   NinBin = np.zeros(Nbins, dtype='int64')

   for n in range(Nbins):

      ind = np.where( ( x > XbinS[n] ) & ( x <= XbinS[n+1] ) )[0]

      NinBin[n] = ind.size

      if NinBin[n] > 0:
        ymed[:,n] = np.percentile( y[ ind ], value )


   return xbin, ymed, NinBin



##############################################################################################
##############################################################################################

def make_increase(input_var):   # don't allow var to decrease with time    
   var = input_var.copy()
   if input_var[0] < input_var[-1]:
     var = var[::-1]
   var_change = var[:-1] - var[1:]
   ind = np.where( var_change < 0 )[0]
   while ind.size > 0:
     var[ind] = var[ind+1]
     var_change = var[:-1] - var[1:]
     ind = np.where( var_change < 0 )[0]
   if input_var[0] < input_var[-1]:
     var = var[::-1]
   return var


def mysmooth(x, N, sfac=2., window='flat'):
    x_smooth = x.copy()
    if x.size < sfac*N:
      print '... mysmooth: input array too small for window size'
      return x
    x_tmp = util.smooth( x, window_len=sfac*N, window=window )
    for i in range(x.size):
       if x[i] > x_tmp[i]:
         x_smooth[i] = x_tmp[i]
    return util.smooth( x_smooth, window_len=N, window=window )


def running_min(x, N, reverse=True):
    if reverse:
      x_min = x.copy()[::-1]
    else:
      x_min = x.copy()
    for i in range(x.size):
       if x.size > i+N:
         x_min[i] = np.min( x[i:i+N])
       else:
         x_min[i] = np.min( x[i:x.size])
    return x_min


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N   # faster than convolution ??


def smooth_box(y, nbox):

   """
   box = np.ones(nbox) / nbox
   y_smooth = np.convolve(y, box, mode='same')
   #correct for boundary effects
   ny = y.size
   y_smooth[0:nbox] = y[0:nbox]
   y_smooth[ny-nbox:ny] = y[ny-nbox:ny]
   """

   ns = y.size
   y_smooth = y.copy()
  
   for j in xrange(nbox,ns-nbox,1):

      #y_smooth[j] = y[j-nbox:j+nbox+1].sum() / ( 2*nbox+1) 
      #y_smooth[j] = y[j-nbox:j+nbox+1].median()
      y_smooth[j] = np.mean( y[j-nbox:j+nbox+1] )



   return y_smooth



def medfilt (x, k):
   """
   Apply a length-k median filter to a 1D array x.
   Boundaries are extended by repeating endpoints.
   """
   assert k % 2 == 1, "Median filter length must be odd."
   assert x.ndim == 1, "Input must be one-dimensional."
   k2 = (k - 1) // 2
   y = np.zeros ((len (x), k), dtype=x.dtype)
   y[:,k2] = x
   for i in range (k2):
      j = k2 - i
      y[j:,i] = x[:-j]
      y[:j,i] = x[0]
      y[:-j,-(i+1)] = x[j:]
      y[-j:,-(i+1)] = x[-1]
   return np.median (y, axis=1)




def mysplit(thestring):

   a, b = string.split( thestring )

   return int(a), int(b)


def split_list(thelist):

   vectorize_mysplit = np.vectorize(mysplit)

   return vectorize_mysplit(thelist)


def MyLine(x, a, b):
   return a + b * x




