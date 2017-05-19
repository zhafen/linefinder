#!/usr/bin/env python
'''Script for categorizing particles into different accretion modes.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
from scipy import stats
import gadget as g
import utilities as util
import pandas as pd
import time as time
import os as os
import sys as sys
import gc as gc
import h5py as h5py
from glob import glob

import astro_tools
from tracking_constants import *

time_start = time.time()




#######################################
 ### DEFINE SIMULATION DIRECTORIES ###
#######################################

sim_list = [ #'m13_mr_Dec16_2013'
             'm12v_mr_Dec5_2013_3'
             #'B1_hr_Dec5_2013_11'
             #'m12qq_hr_Dec16_2013'
             #'m11_hhr_Jan9_2013'
             #'m10_hr_Dec9_2013'
             #, 'm09_hr_Dec16_2013'
           ]

"""
sim_list = [ 'm13_mr_Dec16_2013',
             'm12v_mr_Dec5_2013_3',
             'B1_hr_Dec5_2013_11',
             'm12qq_hr_Dec16_2013',
             'm11_hhr_Jan9_2013',
             'm10_hr_Dec9_2013'
             #, 'm09_hr_Dec16_2013'
           ]
"""

idtag = 'n5'
#idtag = 'all'



GalDef = 2                  # imposed size of SKID galaxy in units of the stellar effective radius 
WindVelMin = 15             # minimum absolute radial velocity to be considered wind in km/s
WindVelMinVc = 2            # minimum radial velocity to be considered wind in units of the maximum circular velocity
ColdHotDef = 2.5e5          # Temperature (k) threshold to separate COLD/HOT modes
TimeMin = 100.              # Minimum time (Myr) spent in other galaxy prior to first accretion to qualify as EXTERNALLY-PROCESSED contribution 
TimeIntervalFac = 5.        # Externally-processed mass is required to spend at least TimeMin during the interval TimeIntervalFac x TimeMin prior to accretion to qualify as MERGER
neg = 5                     # Number of earliest snapshots for which we neglect accretion/ejection events


outname = 'accmode_idlist_%s_g%dv%dvc%dt%dti%dneg%d.hdf5' % ( idtag, GalDef, WindVelMin, WindVelMinVc, TimeMin, TimeIntervalFac, neg )


skipreading = 0


for sname in sim_list:

 simdir = '/scratch/02089/anglesd/FIRE/' + sname + '/'
 outdir = '/work/02089/anglesd/FIRE/' + sname + '/'

 print '\n\n... doing ', sname, ' ...'

 #########################
  ### READ DATA FILES ###
 #########################

 if skipreading == 0:

   if sname[0:3] == 'm13':
     grstr = 'gr1'
   else:
     grstr = 'gr0'

   # --- MAIN galaxy as a function of redshift
   skidname = outdir + 'skidgal_' + grstr + '.hdf5'
   skidgal = h5py.File( skidname, 'r')
   nsgal = skidgal['redshift'][:].size

   # --- particle tracking data
   #ptrack_name = 's440p4n5'
   #f = h5py.File(outdir + 'ptrack_' + ptrack_name + '-' + sname + '.hdf5', 'r')
   ptrack_name = outdir + 'ptrack_idlist_' + idtag + '.hdf5'
   f = h5py.File(ptrack_name, 'r')
   nstrack = f['redshift'][:].size

   # --- header info
   header = g.readsnap( simdir, skidgal['snapnum'][0], 0, header_only=1 )

   npart = len(f['id'])
   nsnap = np.min( [ nsgal, nstrack ] )


 # --- check if redshifts match between particle-halo information
 if np.max( np.abs(f['redshift'][0:nsnap]-skidgal['redshift'][0:nsnap]) ) > 1e-4:
   print '\nWARNING!  redshifts do NOT match!\n'


 print '\nDone with reading.'
 sys.stdout.flush()


 #########################
  ### CALCULATE STUFF ###
 #########################

 # --- coordinates wrt galaxy center (physical kpc)
 r = f['p'][:,0:nsnap,:] - skidgal['p_phi'][0:nsnap,:]

 # --- radial distance to galaxy center (physical kpc)
 R = np.sqrt((r*r).sum(axis=2))                   

 # --- Hubble factor at all redshift
 hubble_factor = astro_tools.hubble_z( f['redshift'][0:nsnap], h=header['hubble'], Omega0=header['Omega0'], OmegaLambda=header['OmegaLambda'] )

 # --- physical velocity wrt galaxy center (km/s)
 # WARNING: need to update with v_phi??
 v = f['v'][:,0:nsnap,:] - skidgal['v_CM'][0:nsnap,:]  +  hubble_factor[:,np.newaxis] * r * UnitLength_in_cm  / UnitVelocity_in_cm_per_s  

 # --- radial velocity wrt galaxy center (km/s)
 Vr = (v*r).sum(axis=2) / R 

 # --- replicate redshifts for indexing (last one removed)
 redshift = np.tile( f['redshift'][0:nsnap], (npart,1) )   

 # --- age of the universe in Myr
 time = 1e3 * util.age_of_universe( redshift, h=header['hubble'], Omega_M=header['Omega0'] )
 dt = time[:,:-1] - time[:,1:] 

 # --- list of snapshots for indexing
 snaplist = skidgal['snapnum'][0:nsnap] 

 # --- index to revert order of redshift snapshots
 ind_rev = np.arange(nsnap-2,-1,-1)  



 print '\nDone with initial calculations.'
 sys.stdout.flush()



 ###############################################
  ### IDENTIFY ACCRETION, EJECTION, MERGERS ###
 ###############################################

 # --- find if particles are inside/outside of main galaxy at each redshift
 IsInGalID = ( f['GalID'][:,0:nsnap] == skidgal['GalID'][0:nsnap] ).astype(int)

 IsInGalRe = ( R < GalDef*skidgal['ReStar'][0:nsnap] ).astype(int)

 IsInOtherGal = ( f['GalID'][:,0:nsnap] > 0 )  &  ( IsInGalID == 0 )
 IsOutsideAnyGal = f['GalID'][:,0:nsnap] <= 0 

 # --- Identify accretion/ejection events relative to main galaxy at each redshift
 #    GalEvent = 0 (no change), 1 (entering galaxy), -1 (leaving galaxy) at that redshift
 GalEventID = IsInGalID[:,0:nsnap-1] - IsInGalID[:,1:nsnap]      

 GalEventRe = IsInGalRe[:,0:nsnap-1] - IsInGalRe[:,1:nsnap]

 # --- identify ALL gas wind ejection events
                # (GalEventID == -1) | (GalEventRe == -1)    DAA: could double count if both conditions are satisfied in subsequent snapshots...
 IsEjected = (  ( GalEventID == -1 )  &
                ( Vr[:,0:nsnap-1] > WindVelMinVc*skidgal['VcMax'][0:nsnap-1] )  &
                ( Vr[:,0:nsnap-1] > WindVelMin )  &
                ( f['Ptype'][:,0:nsnap-1]==0 )  &
                ( IsOutsideAnyGal[:,0:nsnap-1] )  ).astype(int)

 # --- identify ALL gas/star accretion events
 IsAccreted = ( GalEventID == 1 ).astype(int)

 # --- correct for "boundary conditions": neglect events at earliest snapshots
 IsEjected[:,-neg:] = 0
 IsAccreted[:,-neg:] = 0

 Neject = IsEjected.sum(axis=1)
 #Nacc = IsAccreted.sum(axis=1)

 # --- identify FIRST accretion event
 CumNumAcc = IsAccreted[:,ind_rev].cumsum(axis=1)[:,ind_rev]      # cumulative number of ACCRETION events

 IsFirstAcc = IsAccreted  &  ( CumNumAcc == 1 )

 IsJustBeforeFirstAcc = np.roll( IsFirstAcc, 1, axis=1 );   IsJustBeforeFirstAcc[:,0] = 0

 BeforeFirstAcc = ( CumNumAcc == 0 )  &  ( IsInGalID[:,0:nsnap-1] == 0 )

 # --- identify LAST ejection event
 CumNumEject = IsEjected[:,ind_rev].cumsum(axis=1)[:,ind_rev]      # cumulative number of EJECTION events

 CumNumEject_rev = IsEjected.cumsum(axis=1)                           # cumulative number of EJECTION events (in reverse order!)
 IsLastEject = IsEjected  &  ( CumNumEject_rev == 1 )

 # --- find star/gas particles inside the galaxy
 IsStarInside = IsInGalID  &  IsInGalRe  &  ( f['Ptype'][:,0:nsnap]==4 )
 IsGasInside = IsInGalID  &  IsInGalRe  &  ( f['Ptype'][:,0:nsnap]==0 )

 # --- identify STAR FORMATION event inside galaxy
 IsStarFormed = IsStarInside[:,0:nsnap-1]  &  ( f['Ptype'][:,1:nsnap] == 0 )

 # --- identify GAS ACCRETION events
 IsGasAccreted = np.zeros( (npart,nsnap), dtype=np.int32 )
 IsGasFirstAcc = IsGasAccreted.copy()

 IsGasAccreted[:,0:nsnap-1] = IsAccreted  &  ( f['Ptype'][:,0:nsnap-1]==0 )   # initialize with all gas accretion events including "false" events 
 IsGasFirstAcc[:,0:nsnap-1] = IsFirstAcc  &  ( f['Ptype'][:,0:nsnap-1]==0 )   # only the first accretion event

 Nacc = IsGasAccreted.sum(axis=1)

 ind = np.where( Neject == 0 )[0]
 if ind.size > 0:
   IsGasAccreted[ind,:] = IsGasFirstAcc[ind,:]

 ind = np.where( (Nacc - Neject > 1)  &  (Neject > 0) )[0] 

 for i in ind:
    ind_eject = np.where( IsEjected[i,:] == 1 )[0]
    ind_acc = np.where( IsGasAccreted[i,:] == 1 )[0]
    IsGasAccreted[i,:] = IsGasFirstAcc[i,:]                             # initialize only to the first accretion event
    ind_this = np.searchsorted( ind_acc, ind_eject ) - 1
    ind_this = np.unique( ind_this[ ind_this>=0 ] )    
    IsGasAccreted[ i, ind_acc[ind_this] ] = 1
 Nacc = IsGasAccreted.sum(axis=1)


 print '\nDone with indexing.'
 sys.stdout.flush()


##########################
 ### DARK MATTER HALO ###
##########################

 # --- identify the host dark matter halo

 HaloID = np.zeros(nsnap, dtype='int')
 Rvir = np.zeros(nsnap)
 Mvir = np.zeros(nsnap)
 nmaxh = 1000
 for ns in range(snaplist.size):
    halos = astro_tools.read_AHF_halos(simdir, snaplist[ns] )
    ind = np.where(IsStarInside[:,ns]==1)[0]
    if ind.size <= nmaxh:
      mode, count = stats.mode( f['HaloID'][ ind, ns ] )
    else:
      np.random.seed(seed=1234)
      np.random.shuffle(ind)                                                       ###############################################################
      mode, count = stats.mode( f['HaloID'][ np.sort(ind[0:nmaxh]), ns ] )           #######  WARNING CHECK THIS ###################################
    HaloID[ns] = mode[0]                                                           ##############################################################
    ind_halo = np.where( halos['id'] == HaloID[ns] )[0]
    if ind_halo.size != 1:
      print 'halo not found??  ', ns, snaplist[ns]
      continue
    Rvir[ns] = halos['Rvir'][ind_halo[0]]
    Mvir[ns] = halos['Mvir'][ind_halo[0]]
    

 print '\nDone with Dark matter halo.'
 sys.stdout.flush()



#####################################
 ### JUST BEFORE FIRST ACCRETION ###
#####################################

 mask = np.logical_not(IsJustBeforeFirstAcc)     # note that this may not be defined for all particles! 

 # --- redshift at fist accretion onto the galaxy
 redshift_FirstAcc = np.ma.masked_array( redshift[:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

 # --- temperature AT first accretion onto MAIN galaxy
 T_FirstAcc = np.ma.masked_array( f['T'][:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

 # --- metallicity AT first accretion onto MAIN galaxy
 z_FirstAcc = np.ma.masked_array( f['z'][:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

 # --- Ptype at first accretion onto the galaxy
 Ptype_FirstAcc = np.ma.masked_array( f['Ptype'][:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)
 ind = np.where( Ptype_FirstAcc == -1 )[0]
 if ind.size > 0:
   Ptype_FirstAcc[ind] = f['Ptype'][ind,nsnap]           # DAA: check nsnap here!!

 # --- Galaxy ID just prior to first accretion onto the main galaxy
 GalID_FirstAcc = np.ma.masked_array( f['GalID'][:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)


 #######################
  ### LAST EJECTION ###
 #######################

 mask = np.logical_not(IsLastEject)

 # --- redshift at last ejection from the galaxy
 redshift_LastEject = np.ma.masked_array( redshift[:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

 # --- metallicity at last ejection
 z_LastEject = np.ma.masked_array( f['z'][:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)

 print '\nDone with first accretion.'
 sys.stdout.flush()


 ########################
  ### WIND RECYCLING ###
 ########################

 all_DtEject = np.array([])
 all_RmaxEject = np.array([])
 all_RvirEject = np.array([])
 all_RvirEjectIni = np.array([])
 all_ReEject = np.array([])
 all_ReEjectIni = np.array([])
 all_TorbEjectIni = np.array([])
 all_MvirEject = np.array([])
 all_RedshiftEject = np.array([])
 all_RedshiftEjectIni = np.array([])
 all_SnapnumEject = np.array([])

 all_DtAcc = np.array([])
 all_RmaxAcc = np.array([])
 all_RvirAcc = np.array([])
 all_RvirAccIni = np.array([])
 all_ReAcc = np.array([])
 all_ReAccIni = np.array([])
 all_TorbAccIni = np.array([])
 all_MvirAcc = np.array([])
 all_RedshiftAcc = np.array([])
 all_RedshiftAccIni = np.array([])
 all_SnapnumAcc = np.array([])

 for i in range(npart):

    if (Neject[i] < 1) or (Nacc[i] < 1):
      continue

    ind_eject = np.where( IsEjected[i,:] == 1 )[0]
    ind_acc = np.where( IsGasAccreted[i,:] == 1 )[0]
    if (ind_eject.size != Neject[i]) or (ind_acc.size != Nacc[i]):
      print 'ueeeeehhhh!!!'
      continue

    EjectTimes = time[ i, ind_eject ]
    AccTimes = time[ i, ind_acc ]

    # --- from ejection to ejection ---
    if Neject[i] >= 2:
      all_DtEject = np.append( all_DtEject, EjectTimes[:-1] - EjectTimes[1:] )
      all_RedshiftEject = np.append( all_RedshiftEject, redshift[ i, ind_eject[:-1] ] )              # redshift at re-ejection
      all_RedshiftEjectIni = np.append( all_RedshiftEjectIni, redshift[ i, ind_eject[1:] ] )         # redshift at ejection
      all_SnapnumEject = np.append( all_SnapnumEject, snaplist[ ind_eject[:-1] ] )
      for j in range(Neject[i]-1):
         Rmax = np.max( R[ i, ind_eject[j]+1:ind_eject[j+1]+1 ] )
         all_RmaxEject = np.append( all_RmaxEject, Rmax )
         #all_RvirEject = np.append( all_RvirEject, np.mean( Rvir[ind_eject[j]+1:ind_eject[j+1]+1] ) )
         all_RvirEject = np.append( all_RvirEject, Rvir[ind_eject[j]+1] )
         all_RvirEjectIni = np.append( all_RvirEjectIni, Rvir[ind_eject[j+1]] )
         all_ReEject = np.append( all_ReEject, skidgal['ReStar'][ind_eject[j]+1] )
         all_ReEjectIni = np.append( all_ReEjectIni, skidgal['ReStar'][ind_eject[j+1]] )
         all_TorbEjectIni = np.append( all_TorbEjectIni, 2.*np.pi*(skidgal['Rhalf'][ind_eject[j+1]] / skidgal['VcRhalf'][ind_eject[j+1]]) * CM_PER_KPC/CM_PER_KM/SEC_PER_YEAR/1e6 )
         all_MvirEject = np.append( all_MvirEject, np.mean( Mvir[ind_eject[j]+1:ind_eject[j+1]+1] ) )

    # --- from ejection to accretion ---
    if ind_acc[0] < ind_eject[0]:
      all_DtAcc = np.append( all_DtAcc, AccTimes[0:Neject[i]] - EjectTimes[:] )
      all_RedshiftAcc = np.append( all_RedshiftAcc, redshift[ i, ind_acc[0:Neject[i]] ] )    # redshift at re-accretion
      all_RedshiftAccIni = np.append( all_RedshiftAccIni, redshift[ i, ind_eject ] )         # redshift at ejection
      all_SnapnumAcc = np.append( all_SnapnumAcc, snaplist[ ind_acc[0:Neject[i]] ] )
      for j in range(Neject[i]):
         Rmax = np.max( R[ i, ind_acc[j]:ind_eject[j]+1 ] )
         all_RmaxAcc = np.append( all_RmaxAcc, Rmax )
         #all_RvirAcc = np.append( all_RvirAcc, np.mean( Rvir[ind_acc[j]:ind_eject[j]+1] ) )
         all_RvirAcc = np.append( all_RvirAcc, Rvir[ind_acc[j]] )
         all_RvirAccIni = np.append( all_RvirAccIni, Rvir[ind_eject[j]] )
         all_ReAcc = np.append( all_ReAcc, skidgal['ReStar'][ind_acc[j]] )
         all_ReAccIni = np.append( all_ReAccIni, skidgal['ReStar'][ind_eject[j]] )
         all_TorbAccIni = np.append( all_TorbAccIni, 2.*np.pi*(skidgal['Rhalf'][ind_eject[j]] / skidgal['VcRhalf'][ind_eject[j]]) * CM_PER_KPC/CM_PER_KM/SEC_PER_YEAR/1e6 )
         all_MvirAcc = np.append( all_MvirAcc, np.mean( Mvir[ind_acc[j]:ind_eject[j]+1] ) )
    elif Neject[i] >= 2:
      all_DtAcc = np.append( all_DtAcc, AccTimes[0:Neject[i]-1] - EjectTimes[1:] )
      all_RedshiftAcc = np.append( all_RedshiftAcc, redshift[ i, ind_acc[0:Neject[i]-1] ] )   # redshift at re-accretion
      all_RedshiftAccIni = np.append( all_RedshiftAccIni, redshift[ i, ind_eject[1:] ] )      # redshift at ejection
      all_SnapnumAcc = np.append( all_SnapnumAcc, snaplist[ ind_acc[0:Neject[i]-1] ] )
      for j in range(Neject[i]-1):
         Rmax = np.max( R[ i, ind_acc[j]:ind_eject[j+1]+1 ] )
         all_RmaxAcc = np.append( all_RmaxAcc, Rmax )
         #all_RvirAcc = np.append( all_RvirAcc, np.mean( Rvir[ind_acc[j]:ind_eject[j+1]+1] ) )
         all_RvirAcc = np.append( all_RvirAcc, Rvir[ind_acc[j]] )
         all_RvirAccIni = np.append( all_RvirAccIni, Rvir[ind_eject[j+1]] )
         all_ReAcc = np.append( all_ReAcc, skidgal['ReStar'][ind_acc[j]] )
         all_ReAccIni = np.append( all_ReAccIni, skidgal['ReStar'][ind_eject[j+1]] )
         all_TorbAccIni = np.append( all_TorbAccIni, 2.*np.pi*(skidgal['Rhalf'][ind_eject[j+1]] / skidgal['VcRhalf'][ind_eject[j+1]]) * CM_PER_KPC/CM_PER_KM/SEC_PER_YEAR/1e6 )
         all_MvirAcc = np.append( all_MvirAcc, np.mean( Mvir[ind_acc[j]:ind_eject[j+1]+1] ) )

 if all_DtEject.min() < 0:
   print '....... problems with all_DtEject .......'
 if all_DtAcc.min() < 0:
   print '....... problems with all_DtAcc .......'

 print '\nDone with wind recycling.'
 sys.stdout.flush()



 ##################################
  ### SEPARATE ACCRETION MODES ###
 ##################################

 #--- maximum temperature reached OUTSIDE of ANY galaxy  (T values are "invalid" when the particle is inside any galaxy)
 mask = np.logical_not(IsOutsideAnyGal)
 TmaxOutside = np.ma.masked_array( f['T'][:,0:nsnap], mask=mask ).max(axis=1).filled(fill_value =-1)

 #--- maximum temperature reached BEFORE first accretion onto MAIN galaxy and OUTSIDE of any galaxy  (T values are "invalid" after first accretion)
 mask = np.logical_not( BeforeFirstAcc & IsOutsideAnyGal[:,0:nsnap-1] )
 TmaxBeforeAcc = np.ma.masked_array( f['T'][:,0:nsnap-1], mask=mask ).max(axis=1).filled(fill_value =-1)
 
 #--- COLD vs HOT 

 Tmax = TmaxBeforeAcc

 IsHotMode = ( Tmax > ColdHotDef ).astype(int)
 IsColdMode = ( Tmax <= ColdHotDef ).astype(int)

 IsHotMode_mask = np.tile( IsHotMode[:,np.newaxis], nsnap )
 IsColdMode_mask = np.tile( IsColdMode[:,np.newaxis], nsnap )

 #--- WIND vs NO-WIND
 # IsWind[i,n]=YES for particle i if it has been ejected at least once before snapshot n 

 IsWind = np.zeros( (npart,nsnap), dtype=np.int32 )
 IsWind[:,0:nsnap-1] = ( CumNumEject >= 1 ).astype(int)         

 #correct for "boundary conditions": particles inside galaxy at earliest snapshots cannot count as winds
 #for k in range(4):
 #   IsWind[ IsInGalID[:,nsnap-1-k] == 1, nsnap-1-k ] = 0



 ###################################################
  ### TOTAL STELLAR MASS IN EACH ACCRETION MODE ###
 ###################################################

 Nstar = IsStarInside.sum(axis=0)

 StarMass = ( f['m'][:,0:nsnap] * IsStarInside ).sum(axis=0)
 StarMassHot = ( f['m'][:,0:nsnap] * IsStarInside * IsHotMode_mask ).sum(axis=0)
 StarMassCold = ( f['m'][:,0:nsnap] * IsStarInside * IsColdMode_mask ).sum(axis=0)
 StarMassWind = ( f['m'][:,0:nsnap] * IsStarInside * IsWind ).sum(axis=0)
 StarMassWindHot = ( f['m'][:,0:nsnap] * IsStarInside * IsWind * IsHotMode_mask ).sum(axis=0)
 StarMassWindCold = ( f['m'][:,0:nsnap] * IsStarInside * IsWind * IsColdMode_mask ).sum(axis=0)

 GasMass = ( f['m'][:,0:nsnap] * IsGasInside ).sum(axis=0)
 GasMassWind = ( f['m'][:,0:nsnap] * IsGasInside * IsWind ).sum(axis=0)

 Sfr = ( f['sfr'][:,0:nsnap] * IsGasInside ).sum(axis=0)
 SfrHot = ( f['sfr'][:,0:nsnap] * IsGasInside * IsHotMode_mask ).sum(axis=0)
 SfrCold = ( f['sfr'][:,0:nsnap] * IsGasInside * IsColdMode_mask ).sum(axis=0)
 SfrWind = ( f['sfr'][:,0:nsnap] * IsGasInside * IsWind ).sum(axis=0)
 SfrWindHot = ( f['sfr'][:,0:nsnap] * IsGasInside * IsWind * IsHotMode_mask ).sum(axis=0)
 SfrWindCold = ( f['sfr'][:,0:nsnap] * IsGasInside * IsWind * IsColdMode_mask ).sum(axis=0)

 # --- "Msun per snapshot" from gas accretion events
 AccretedGasMass = ( f['m'][:,0:nsnap] * IsGasAccreted ).sum(axis=0)
 AccretedGasMassWind = ( f['m'][:,0:nsnap] * IsGasAccreted * IsWind ).sum(axis=0) 



 ###############################
  ### SEPARATE GROWTH MODES ###
 ###############################

 TimeInOtherGal = ( dt * IsInOtherGal[:,0:nsnap-1].astype(int) ).sum(axis=1)
 TimeInOtherGalBeforeAcc = ( dt * (BeforeFirstAcc & IsInOtherGal[:,0:nsnap-1]).astype(int) ).sum(axis=1)
 #TimeInOtherGalBeforeAccAsStar = ( dt * (BeforeFirstAcc & IsInOtherGal[:,0:nsnap-1] & (f['Ptype'][:,0:nsnap-1]==4)).astype(int) ).sum(axis=1)

 CumTimeBeforeAcc = ( dt * BeforeFirstAcc.astype(int) ).cumsum(axis=1)
 IsTimeIntervalBeforeAcc = ( (CumTimeBeforeAcc <= TimeIntervalFac*TimeMin) & BeforeFirstAcc ).astype(int)
 TimeInOtherGalBeforeAccDuringInterval = ( dt * (IsTimeIntervalBeforeAcc & IsInOtherGal[:,0:nsnap-1]).astype(int) ).sum(axis=1)


 #--- GrowthMode = PRISTINE      --> "non-externally processed"  
 #                 PRE-PROCESSED --> "externally processed"

 IsPristine = ( TimeInOtherGalBeforeAcc < TimeMin ).astype(int)
 IsPreProcessed = ( TimeInOtherGalBeforeAcc >= TimeMin ).astype(int)
 #correct "boundary conditions": particles inside galaxy at earliest snapshot count as pristine
 for k in range(neg):
    IsPristine[ IsInGalID[:,nsnap-1-k] == 1 ] = 1         
    IsPreProcessed[ IsInGalID[:,nsnap-1-k] == 1 ] = 0

 #--- PRE-PROCESSED GrowthMode: MERGER vs MASS TRANSFER

 IsMassTransfer = (  IsPreProcessed & (TimeInOtherGalBeforeAccDuringInterval < TimeMin) ).astype(int)
 IsMerger = (  IsPreProcessed & (TimeInOtherGalBeforeAccDuringInterval >= TimeMin)  ).astype(int)



 ############################################
  ### all specific to PRE-PROCESSED MODE ###
 ############################################

 IsPreProcessed_mask = np.tile( IsPreProcessed[:,np.newaxis], nsnap )
 IsMassTransfer_mask = np.tile( IsMassTransfer[:,np.newaxis], nsnap )
 IsMerger_mask = np.tile( IsMerger[:,np.newaxis], nsnap )
 IsStarAcc_mask = np.tile( (Ptype_FirstAcc == 4).astype(int)[:,np.newaxis], nsnap )
 IsGasAcc_mask = np.tile( (Ptype_FirstAcc == 0).astype(int)[:,np.newaxis], nsnap )

 StarMassFromPreProcessed = ( f['m'][:,0:nsnap] * IsStarInside * IsPreProcessed_mask ).sum(axis=0)
 StarMassFromMassTransfer = ( f['m'][:,0:nsnap] * IsStarInside * IsMassTransfer_mask ).sum(axis=0)
 StarMassFromMassTransferGas = ( f['m'][:,0:nsnap] * IsStarInside * IsMassTransfer_mask * IsGasAcc_mask ).sum(axis=0)
 StarMassFromMassTransferStar = ( f['m'][:,0:nsnap] * IsStarInside * IsMassTransfer_mask * IsStarAcc_mask ).sum(axis=0)
 StarMassFromMerger = ( f['m'][:,0:nsnap] * IsStarInside * IsMerger_mask ).sum(axis=0)
 StarMassFromMergerGas = ( f['m'][:,0:nsnap] * IsStarInside * IsMerger_mask * IsGasAcc_mask ).sum(axis=0)
 StarMassFromMergerStar = ( f['m'][:,0:nsnap] * IsStarInside * IsMerger_mask * IsStarAcc_mask ).sum(axis=0)

 GasMassFromPreProcessed = ( f['m'][:,0:nsnap] * IsGasInside * IsPreProcessed_mask ).sum(axis=0)
 GasMassFromMassTransfer = ( f['m'][:,0:nsnap] * IsGasInside * IsMassTransfer_mask ).sum(axis=0)
 GasMassFromMerger = ( f['m'][:,0:nsnap] * IsGasInside * IsMerger_mask ).sum(axis=0)

 SfrFromPreProcessed = ( f['sfr'][:,0:nsnap] * IsGasInside * IsPreProcessed_mask ).sum(axis=0) 
 SfrFromMassTransfer = ( f['sfr'][:,0:nsnap] * IsGasInside * IsMassTransfer_mask ).sum(axis=0)
 SfrFromMerger = ( f['sfr'][:,0:nsnap] * IsGasInside * IsMerger_mask ).sum(axis=0)

 # --- "Msun per snapshot" from gas accretion events
 AccretedGasMassFromPreProcessed = ( f['m'][:,0:nsnap] * IsGasAccreted * IsPreProcessed_mask ).sum(axis=0)  
 AccretedGasMassFromTransfer = ( f['m'][:,0:nsnap] * IsGasAccreted * IsMassTransfer_mask ).sum(axis=0)
 AccretedGasMassFromMerger = ( f['m'][:,0:nsnap] * IsGasAccreted * IsMerger_mask ).sum(axis=0)


 #######################################
  ### all specific to PRISTINE MODE ###
 #######################################

 IsPristine_mask = np.tile( IsPristine[:,np.newaxis], nsnap )

 StarMassFromPristine = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask ).sum(axis=0)
 StarMassFromPristineHot = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsHotMode_mask ).sum(axis=0)
 StarMassFromPristineCold = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsColdMode_mask ).sum(axis=0)
 StarMassFromPristineWind = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsWind ).sum(axis=0)
 StarMassFromPristineWindHot = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsWind * IsHotMode_mask ).sum(axis=0)
 StarMassFromPristineWindCold = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsWind * IsColdMode_mask ).sum(axis=0)

 StarMassFromPristineGas = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsGasAcc_mask ).sum(axis=0)
 StarMassFromPristineHotGas = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsHotMode_mask * IsGasAcc_mask ).sum(axis=0)
 StarMassFromPristineColdGas = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsColdMode_mask * IsGasAcc_mask ).sum(axis=0)
 StarMassFromPristineWindGas = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsWind * IsGasAcc_mask ).sum(axis=0)
 StarMassFromPristineWindHotGas = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsWind * IsHotMode_mask * IsGasAcc_mask ).sum(axis=0)
 StarMassFromPristineWindColdGas = ( f['m'][:,0:nsnap] * IsStarInside * IsPristine_mask * IsWind * IsColdMode_mask * IsGasAcc_mask ).sum(axis=0)

 GasMassFromPristine = ( f['m'][:,0:nsnap] * IsGasInside * IsPristine_mask ).sum(axis=0)
 GasMassFromPristineWind = ( f['m'][:,0:nsnap] * IsGasInside * IsPristine_mask * IsWind ).sum(axis=0)

 SfrFromPristine = ( f['sfr'][:,0:nsnap] * IsGasInside * IsPristine_mask ).sum(axis=0)
 SfrFromPristineHot = ( f['sfr'][:,0:nsnap] * IsGasInside * IsPristine_mask * IsHotMode_mask ).sum(axis=0)
 SfrFromPristineCold = ( f['sfr'][:,0:nsnap] * IsGasInside * IsPristine_mask * IsColdMode_mask ).sum(axis=0)
 SfrFromPristineWind = ( f['sfr'][:,0:nsnap] * IsGasInside * IsPristine_mask * IsWind ).sum(axis=0)
 SfrFromPristineWindHot = ( f['sfr'][:,0:nsnap] * IsGasInside * IsPristine_mask * IsWind * IsHotMode_mask ).sum(axis=0)
 SfrFromPristineWindCold = ( f['sfr'][:,0:nsnap] * IsGasInside * IsPristine_mask * IsWind * IsColdMode_mask ).sum(axis=0)

 # --- "Msun per snapshot" from gas accretion events
 AccretedGasMassFromPristine = ( f['m'][:,0:nsnap] * IsGasAccreted * IsPristine_mask ).sum(axis=0)
 AccretedGasMassFromPristineWind = ( f['m'][:,0:nsnap] * IsGasAccreted * IsPristine_mask * IsWind ).sum(axis=0)



 ####################################
  ### MASS LOADINGS -- MASS LOST ###
 ####################################

 IsLost = np.tile( (IsInGalID[:,0]==0).astype(int)[:,np.newaxis], nsnap-1 )

 GasMassLost = ( f['m'][:,0:nsnap-1] * IsLastEject * IsLost ).sum(axis=0)

 MetalMassLost = ( f['m'][:,0:nsnap-1] * f['z'][:,0:nsnap-1] * IsLastEject * IsLost ).sum(axis=0) * SolarAbundance

 GasMassEjected = ( f['m'][:,0:nsnap-1] * IsEjected  ).sum(axis=0)

 StarMassFormed = ( f['m'][:,0:nsnap-1] * IsStarFormed ).sum(axis=0)


 print '\nDone with accretion modes.'
 sys.stdout.flush()


 ##############################
  ### SAVE RESULTS TO FILE ###
 ##############################

 #outname = 'accmode_' + ptrack_name + '-' + sname + '.hdf5'
 #outname = 'accmode_idlist_' + idtag + '.hdf5'

 if os.path.isfile(outdir + outname):
   os.remove(outdir + outname)

 outf = h5py.File(outdir + outname, 'w')

 #outf.create_dataset('redshift', data=f['redshift'][0:nsnap])
 outf.create_dataset('redshift', data=redshift[0,:])
 outf.create_dataset('dt', data=dt[0,:])

 outf.create_dataset('R', data=R)
 outf.create_dataset('Vr', data=Vr)

 outf.create_dataset('IsInGalID', data=IsInGalID)
 outf.create_dataset('IsInGalRe', data=IsInGalRe)
 outf.create_dataset('IsStarInside', data=IsStarInside)
 outf.create_dataset('IsGasInside', data=IsGasInside)
 outf.create_dataset('IsEjected', data=IsEjected)
 outf.create_dataset('IsAccreted', data=IsAccreted)
 outf.create_dataset('IsGasAccreted', data=IsGasAccreted)

 outf.create_dataset('IsPristine', data=IsPristine)
 outf.create_dataset('IsPreProcessed', data=IsPreProcessed)
 outf.create_dataset('IsMassTransfer', data=IsMassTransfer)
 outf.create_dataset('IsMerger', data=IsMerger)
 outf.create_dataset('IsHotMode', data=IsHotMode)
 outf.create_dataset('IsColdMode', data=IsColdMode)
 outf.create_dataset('IsWind', data=IsWind)


 outf.create_dataset('TmaxOutside', data=TmaxOutside)
 outf.create_dataset('TmaxBeforeAcc', data=TmaxBeforeAcc)
 outf.create_dataset('Neject', data=Neject)
 outf.create_dataset('Nacc', data=Nacc)
 outf.create_dataset('TimeInOtherGal', data=TimeInOtherGal)
 outf.create_dataset('TimeInOtherGalBeforeAcc', data=TimeInOtherGalBeforeAcc)
 outf.create_dataset('TimeInOtherGalBeforeAccDuringInterval', data=TimeInOtherGalBeforeAccDuringInterval)

 outf.create_dataset('T_FirstAcc', data=T_FirstAcc)
 outf.create_dataset('redshift_FirstAcc', data=redshift_FirstAcc)
 outf.create_dataset('z_FirstAcc', data=z_FirstAcc)
 outf.create_dataset('Ptype_FirstAcc', data=Ptype_FirstAcc)
 outf.create_dataset('GalID_FirstAcc', data=GalID_FirstAcc)

 outf.create_dataset('redshift_LastEject', data=redshift_LastEject)
 outf.create_dataset('z_LastEject', data=z_LastEject)

 outf.create_dataset('all_DtEject', data=all_DtEject)
 outf.create_dataset('all_RmaxEject', data=all_RmaxEject)
 outf.create_dataset('all_RvirEject', data=all_RvirEject)
 outf.create_dataset('all_RvirEjectIni', data=all_RvirEjectIni)
 outf.create_dataset('all_ReEject', data=all_ReEject)
 outf.create_dataset('all_ReEjectIni', data=all_ReEjectIni)
 outf.create_dataset('all_TorbEjectIni', data=all_TorbEjectIni)
 outf.create_dataset('all_MvirEject', data=all_MvirEject)
 outf.create_dataset('all_RedshiftEject', data=all_RedshiftEject)
 outf.create_dataset('all_RedshiftEjectIni', data=all_RedshiftEjectIni)
 outf.create_dataset('all_SnapnumEject', data=all_SnapnumEject)
 outf.create_dataset('all_DtAcc', data=all_DtAcc)
 outf.create_dataset('all_RmaxAcc', data=all_RmaxAcc)
 outf.create_dataset('all_RvirAcc', data=all_RvirAcc)
 outf.create_dataset('all_RvirAccIni', data=all_RvirAccIni)
 outf.create_dataset('all_ReAcc', data=all_ReAcc)
 outf.create_dataset('all_ReAccIni', data=all_ReAccIni)
 outf.create_dataset('all_TorbAccIni', data=all_TorbAccIni)
 outf.create_dataset('all_MvirAcc', data=all_MvirAcc)
 outf.create_dataset('all_RedshiftAcc', data=all_RedshiftAcc)
 outf.create_dataset('all_RedshiftAccIni', data=all_RedshiftAccIni)
 outf.create_dataset('all_SnapnumAcc', data=all_SnapnumAcc)

 outf.create_dataset('Nstar', data=Nstar)

 outf.create_dataset('StarMass', data=StarMass) 
 outf.create_dataset('StarMassHot', data=StarMassHot)
 outf.create_dataset('StarMassCold', data=StarMassCold)
 outf.create_dataset('StarMassWind', data=StarMassWind)
 outf.create_dataset('StarMassWindHot', data=StarMassWindHot)
 outf.create_dataset('StarMassWindCold', data=StarMassWindCold)

 outf.create_dataset('GasMass', data=GasMass)
 outf.create_dataset('GasMassWind', data=GasMassWind)

 outf.create_dataset('Sfr', data=Sfr)
 outf.create_dataset('SfrHot', data=SfrHot)
 outf.create_dataset('SfrCold', data=SfrCold)
 outf.create_dataset('SfrWind', data=SfrWind)
 outf.create_dataset('SfrWindHot', data=SfrWindHot)
 outf.create_dataset('SfrWindCold', data=SfrWindCold)

 outf.create_dataset('StarMassFromPreProcessed', data=StarMassFromPreProcessed)
 outf.create_dataset('StarMassFromMassTransfer', data=StarMassFromMassTransfer)
 outf.create_dataset('StarMassFromMassTransferGas', data=StarMassFromMassTransferGas)
 outf.create_dataset('StarMassFromMassTransferStar', data=StarMassFromMassTransferStar)
 outf.create_dataset('StarMassFromMerger', data=StarMassFromMerger)
 outf.create_dataset('StarMassFromMergerGas', data=StarMassFromMergerGas) 
 outf.create_dataset('StarMassFromMergerStar', data=StarMassFromMergerStar)

 outf.create_dataset('GasMassFromPreProcessed', data=GasMassFromPreProcessed)
 outf.create_dataset('GasMassFromMassTransfer', data=GasMassFromMassTransfer)
 outf.create_dataset('GasMassFromMerger', data=GasMassFromMerger)

 outf.create_dataset('SfrFromPreProcessed', data=SfrFromPreProcessed)
 outf.create_dataset('SfrFromMassTransfer', data=SfrFromMassTransfer)
 outf.create_dataset('SfrFromMerger', data=SfrFromMerger)

 outf.create_dataset('StarMassFromPristine', data=StarMassFromPristine)
 outf.create_dataset('StarMassFromPristineHot', data=StarMassFromPristineHot)
 outf.create_dataset('StarMassFromPristineCold', data=StarMassFromPristineCold)
 outf.create_dataset('StarMassFromPristineWind', data=StarMassFromPristineWind)
 outf.create_dataset('StarMassFromPristineWindHot', data=StarMassFromPristineWindHot)
 outf.create_dataset('StarMassFromPristineWindCold', data=StarMassFromPristineWindCold)

 outf.create_dataset('StarMassFromPristineGas', data=StarMassFromPristineGas)
 outf.create_dataset('StarMassFromPristineHotGas', data=StarMassFromPristineHotGas)
 outf.create_dataset('StarMassFromPristineColdGas', data=StarMassFromPristineColdGas)
 outf.create_dataset('StarMassFromPristineWindGas', data=StarMassFromPristineWindGas)
 outf.create_dataset('StarMassFromPristineWindHotGas', data=StarMassFromPristineWindHotGas)
 outf.create_dataset('StarMassFromPristineWindColdGas', data=StarMassFromPristineWindColdGas)

 outf.create_dataset('GasMassFromPristine', data=GasMassFromPristine)
 outf.create_dataset('GasMassFromPristineWind', data=GasMassFromPristineWind)

 outf.create_dataset('SfrFromPristine', data=SfrFromPristine)            
 outf.create_dataset('SfrFromPristineHot', data=SfrFromPristineHot)
 outf.create_dataset('SfrFromPristineCold', data=SfrFromPristineCold)
 outf.create_dataset('SfrFromPristineWind', data=SfrFromPristineWind)
 outf.create_dataset('SfrFromPristineWindHot', data=SfrFromPristineWindHot)
 outf.create_dataset('SfrFromPristineWindCold', data=SfrFromPristineWindCold)

 outf.create_dataset('AccretedGasMass', data=AccretedGasMass)
 outf.create_dataset('AccretedGasMassWind', data=AccretedGasMassWind)
 outf.create_dataset('AccretedGasMassFromPreProcessed', data=AccretedGasMassFromPreProcessed)
 outf.create_dataset('AccretedGasMassFromTransfer', data=AccretedGasMassFromTransfer)
 outf.create_dataset('AccretedGasMassFromMerger', data=AccretedGasMassFromMerger)
 outf.create_dataset('AccretedGasMassFromPristine', data=AccretedGasMassFromPristine)
 outf.create_dataset('AccretedGasMassFromPristineWind', data=AccretedGasMassFromPristineWind)

 outf.create_dataset('GasMassLost', data=GasMassLost)
 outf.create_dataset('MetalMassLost', data=MetalMassLost)
 outf.create_dataset('GasMassEjected', data=GasMassEjected)
 outf.create_dataset('StarMassFormed', data=StarMassFormed)

 outf.create_dataset('HaloID', data=HaloID)
 outf.create_dataset('Rvir', data=Rvir)
 outf.create_dataset('Mvir', data=Mvir)

 outf.close()

print '\nDone!'



