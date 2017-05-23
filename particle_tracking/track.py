#!/usr/bin/env python
'''Script for tracking particles.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import gadget as g
import pandas as pd
import time as time
import os as os
import sys as sys
import gc as gc
import h5py as h5py

import tracking_tools
from tracking_constants import *

time_start = time.time()




#######################################
 ### DEFINE SIMULATION DIRECTORIES ###
#######################################

# --- check for input parameters
narg = len(sys.argv)
if narg == 3:
   sname = sys.argv[1]
   ntrack = float(sys.argv[2])     # number of particles that we track
else:
   print '-----\n --- using default parameters --- \n-----'
   sname = 'm13_mr_Dec16_2013'
   #sname = 'm12v_mr_Dec5_2013_3'
   #sname = 'm11_hhr_Jan9_2013'
   ntrack = 1e5                


simdir = '/scratch/02089/anglesd/FIRE/' + sname 
outdir = '/work/02089/anglesd/FIRE/' + sname 
if not os.path.exists(outdir):
  os.mkdir(outdir)



#####################################################
 ### DEFINE PARAMETERS AND READ PARTICLE ID LIST ###
#####################################################

# The "Central" galaxy is the most massive galaxy in all runs but m13...  
if sname[0:3] == 'm13':
  grstr = 'gr1' 
else:
  grstr = 'gr0'

idlist = h5py.File( simdir + '/skid/progen_idlist_' + grstr + '.hdf5', 'r')

Ptype = [ 0, 4 ]                     # must contain all possible particle types in idlist

snap_end = 440                       # z = 0
#snap_end = 15

snap_ini = 0

nskip = 0
#nskip = 215


if idlist['id'].size > ntrack:
   ind_myids = np.arange(idlist['id'].size)
   np.random.seed(seed=1234)
   np.random.shuffle(ind_myids)
   theIDs = np.unique( idlist['id'][:][ind_myids[0:ntrack]] )
   tag = 'n{0:1.0f}'.format(np.log10(ntrack))
else:
   theIDs = np.unique( idlist['id'][:] )
   tag = 'all'

idlist.close()



##########################################
 ### LOOP OVER ALL REDSHIFT SNAPSHOTS ###
##########################################


# --- define structure to hold particle tracks ---
# --> ptrack ['varname'] [particle i, snap j, k component]

nsnap = 1 + snap_end - snap_ini       # number of redshift snapshots that we follow back
ntrack = theIDs.size

#myfloat = 'float64'
myfloat = 'float32'

ptrack = { 'redshift':np.zeros(nsnap,dtype=myfloat), 
           'snapnum':np.zeros(nsnap,dtype='int16'),
           'id':np.zeros(ntrack,dtype='int64'), 
           'Ptype':np.zeros(ntrack,dtype=('int8',(nsnap,))),
           'rho':np.zeros(ntrack,dtype=(myfloat,(nsnap,))), 
           'sfr':np.zeros(ntrack,dtype=(myfloat,(nsnap,))),
           'T':np.zeros(ntrack,dtype=(myfloat,(nsnap,))),
           'z':np.zeros(ntrack,dtype=(myfloat,(nsnap,))),
           'm':np.zeros(ntrack,dtype=(myfloat,(nsnap,))),
           'p':np.zeros(ntrack,dtype=(myfloat,(nsnap,3))),
           'v':np.zeros(ntrack,dtype=(myfloat,(nsnap,3))), 
           'GalID':np.zeros(ntrack,dtype=('int32',(nsnap,))),
           'HaloID':np.zeros(ntrack,dtype=('int32',(nsnap,))),
           'SubHaloID':np.zeros(ntrack,dtype=('int32',(nsnap,))) }


ptrack['id'][:] = theIDs


print '\n**********************************************************************************'
print simdir, '   ntrack =', ntrack, '   -->  ', tag
print '**********************************************************************************'


j = 0

for ns in range( snap_end, snap_ini-1, -(1+nskip) ):


   time_1 = time.time()

   dfid, redshift = tracking_tools.find_ids( simdir, ns, Ptype, theIDs, HostGalaxy=1, HostHalo=1 )


   #ptrack['redshift'][:,j] = redshift
   ptrack['redshift'][j] = redshift

   ptrack['snapnum'][j] = ns

   ptrack['Ptype'][:,j] = dfid['Ptype'].values

   ptrack['rho'][:,j] = dfid['rho'].values                                                           # cm^(-3)

   ptrack['sfr'][:,j] = dfid['sfr'].values                                                           # Msun / year   (stellar Age in Myr for star particles)

   ptrack['T'][:,j] = dfid['T'].values                                                               # Kelvin

   ptrack['z'][:,j] = dfid['z'].values                                                               # Zsun (metal mass fraction in Solar units)

   ptrack['m'][:,j] = dfid['m'].values                                                               # Msun (particle mass in solar masses)

   ptrack['p'][:,j,:] = np.array( [ dfid['x0'].values, dfid['x1'].values, dfid['x2'].values ] ).T    # kpc (physical)

   ptrack['v'][:,j,:] = np.array( [ dfid['v0'].values, dfid['v1'].values, dfid['v2'].values ] ).T    # km/s (peculiar - need to add H(a)*r contribution)

   ptrack['GalID'][:,j] = dfid['GalID'].values

   ptrack['HaloID'][:,j] = dfid['HaloID'].values

   ptrack['SubHaloID'][:,j] = dfid['SubHaloID'].values


   j += 1

   gc.collect()          # helps stop leaking memory ?
   time_2 = time.time()
   print '\n', ns, ' P[redshift] = ' + '%.3f' % redshift, '    done in ... ', time_2 - time_1, ' seconds'
   print '------------------------------------------------------------------------------------------------\n'
   sys.stdout.flush()



#######################################
 ### WRITE PARTICLE TRACKS TO FILE ###
#######################################


outname = 'ptrack_idlist_' + tag + '.hdf5'


if os.path.isfile(outdir + '/' + outname):
  os.remove(outdir + '/' + outname)

f = h5py.File(outdir + '/' + outname, 'w')
for keyname in ptrack.keys():
    f.create_dataset(keyname, data=ptrack[keyname])
f.close()


time_end = time.time()


print '\n ' + outname + ' ... done in just ', time_end - time_start, ' seconds!'
print '\n ...', (time_end - time_start) / ntrack, ' seconds per particle!\n'






