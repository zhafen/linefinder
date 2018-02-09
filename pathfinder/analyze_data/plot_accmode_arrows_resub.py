# run as >>> execfile('plot_accmode_arrows_resub.py')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cl
import numpy as np
import gadget as g
import h5py as h5py
import time as time
import os as os
import sys as sys
from glob import glob

import daa_lib as daa
from daa_constants import *

import imp
imp.reload(daa)


sim_list = [ 'm13_mr_Dec16_2013',
             'm12v_mr_Dec5_2013_3',
             'B1_hr_Dec5_2013_11',
             'm12qq_hr_Dec16_2013',
             'm11_hhr_Jan9_2013',
             'm10_hr_Dec9_2013'
             #, 'm09_hr_Dec16_2013'
           ]


isim = 4
#idtag = 'n5'
idtag = 'all'


ntest = 5000                 # number of trajectories to plot
ntestS = 1000                # number of star particles to plot
npix = 256
nstp = 2                      # number of snapshot to include ejection events
nsmooth = 20                  # for plotting trajectories
navg = 5                      # for velocity relative to central galaxy

if isim == 0:
 xrange = [-300,300]          #extent of plot axes in kpc
 yrange = [-300,300]
elif isim == 1:
 xrange = [-500,200]          #extent of plot axes in kpc
 yrange = [-300,400]
elif isim == 4:
 xrange = [-120,120]          #extent of plot axes in kpc
 yrange = [-120,120]
else:
 xrange = [-200,200]          #extent of plot axes in kpc
 yrange = [-200,200]


skipreading = 1


sname = sim_list[isim]

simdir = '/projects/b1026/xvault/anglesd/FIRE/' + sname + '/'

#outdir = './accmode/' + sname + '/arrow/'
outdir = './accmode_resub/'

if not os.path.exists(outdir):
  try:
    os.mkdir(outdir)
  except OSError:
    try:
      os.makedirs(outdir)
    except OSError:
      print '...could not create output directory: ' + outdir



#--- plotting parameters

np.random.seed(seed=1234)

rasterized = False

acirc = np.linspace(0,2*np.pi,100)
xcirc = np.cos(acirc)
ycirc = np.sin(acirc)

lw = 1

header = g.readsnap(simdir,440,0,cosmological=1,skip_bh=1,header_only=1)
h = header['hubble']

if skipreading==0:

 # --- read main galaxy info
 if sname[0:3] == 'm13':
   grstr = 'gr1'
 else:
   grstr = 'gr0'
 skidname = simdir + 'skidgal_' + grstr + '.hdf5'
 skidgal = h5py.File( skidname, 'r')

 # --- read ptrack file
 ptrack_name = simdir + 'ptrack_idlist_' + idtag + '.hdf5'
 ptr = h5py.File(ptrack_name, 'r')

 # --- read accmode file
 GalDef = 2
 WindVelMin = 15
 WindVelMinVc = 2
 TimeMin = 100.
 TimeIntervalFac = 5.
 neg = 5
 accname = 'accmode_idlist_%s_g%dv%dvc%dt%dti%dneg%d.hdf5' % ( idtag, GalDef, WindVelMin, WindVelMinVc, TimeMin, TimeIntervalFac, neg )
 f = h5py.File( simdir + accname, 'r')

 nsnap = f['redshift'][:].size
 snap_list = skidgal['snapnum'][0:nsnap]

 # --- coordinates wrt galaxy center (physical kpc)
 z = ptr['redshift'][0:nsnap]
 hubble_factor = daa.hubble_z( z ) #, h=header['hubble'], Omega0=header['Omega0'], OmegaLambda=header['OmegaLambda'] )
 r = ptr['p'][:,0:nsnap,:] - skidgal['p_phi'][0:nsnap,:]
 R = np.sqrt((r*r).sum(axis=2))
 v = ptr['v'][:,0:nsnap,:] - skidgal['v_CM'][0:nsnap,:]  +  hubble_factor[:,np.newaxis] * r * UnitLength_in_cm  / UnitVelocity_in_cm_per_s


 #IsInGalID = ( ptr['GalID'][:,0:nsnap] == skidgal['GalID'][0:nsnap] ).astype(int)
 ind_rev = np.arange(nsnap-2,-1,-1)
 IsInGalID = f['IsInGalID'][:,:]
 IsInOtherGal = ( (ptr['GalID'][:,0:nsnap] > 0)  &  (IsInGalID == 0) ).astype(int)
 IsAfterOtherGal = ( IsInOtherGal.cumsum(axis=1) == 0 ).astype(int)
 IsGasAccreted = f['IsGasAccreted'][:,:]
 IsEjected = f['IsEjected'][:,:]
 CumNumEject = IsEjected[:,ind_rev].cumsum(axis=1)[:,ind_rev]      # cumulative number of EJECTION events
 IsFirstEject = ( IsEjected  &  ( CumNumEject == 1 ) ).astype(int)


for i in range(nsnap):

   snapnum = snap_list[i]
   redshift = ptr['redshift'][i]

   #if round(redshift,3) not in [ 4., 3.5, 3., 2.5, 2., 1.5 ]:
   #  continue

   if round(redshift,3) not in [ 2.9, 2.4, 2.0, 1.9, 1.51, 1.48, 1.06, 0.82, 0.47 ]:
   #if round(redshift,3) not in [ 2.9 ]:
     continue


   # --- FIGURE ---   
   #figname = 'arr_' + g.snap_ext(snapnum) + '.pdf'
   figname = 'arr_s' + str(nstp) + '_' + g.snap_ext(snapnum) + '.png'
   fig = plt.figure( figsize=(4,4) )
   fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
   ax = fig.add_subplot(111)
   if rasterized:
        ax.set_rasterization_zorder(1)
   for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
     item.set_fontsize(10)
   ax.set_aspect('equal')
   if round(redshift,3) == 1.48:
     ax.text( 0.02, 0.94, 'z = '+ '%.2f' % (redshift), {'color': 'k', 'fontsize': 14}, transform=ax.transAxes ) 
   else:
     ax.text( 0.02, 0.94, 'z = '+ '%.1f' % (redshift), {'color': 'k', 'fontsize': 14}, transform=ax.transAxes )
   ax.set_xlim(xrange[0],xrange[1])  
   ax.set_ylim(yrange[0],yrange[1]) 
   for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(1.3)
   ax.xaxis.set_tick_params(width=1.3)
   ax.yaxis.set_tick_params(width=1.3)
   mew = 1



   
   # --- density grid
   P = g.readsnap( simdir, snapnum, 0, cosmological=1 )
   x = P['p'][:,0] - skidgal['p_phi'][i,0]
   y = P['p'][:,1] - skidgal['p_phi'][i,1]
   m = P['m'][:]
   hsml = P['h'][:]
   m_map = daa.make_2Dgrid( x, y, xrange, yrange=yrange, weight1=m, hsml=hsml, pixels=npix )
   

   ind_gas = np.where( ptr['Ptype'][:,i] == 0 )[0]
   """
   m = ptr['m'][:,i]
   rho = (4./3.) * np.pi * ptr['rho'][:,i]
   x = r[:,i,0]  #ptr['p'][:,i,0] - skidgal['p_phi'][i,0]
   y = r[:,i,1]
   hsml = (m[ind_gas]/rho[ind_gas])**(1./3) * ((MSUN/PROTONMASS)**(1./3)/CM_PER_KPC)
   m_map = daa.make_2Dgrid( x[ind_gas], y[ind_gas], xrange, yrange=yrange, weight1=m[ind_gas], hsml=hsml, pixels=npix )
   """

   m_map = daa.clip_2Dgrid( m_map )
   m_map /= m_map.max()
   # --- plot 2D grid
   #vmin = -4
   #vmax = 0
   #vmin = np.log10(m_map.T).min() - 1.5 #2.
   m_map[m_map<1e-4] = 1e-4
   vmin = np.log10(m_map.T).max() - 4.5
   vmax = np.log10(m_map.T).max() - 1
   im = ax.imshow( np.log10( m_map.T ), vmin=vmin, vmax=vmax, cmap=cm.Greys, interpolation='bicubic', origin='low', extent=[xrange[0],xrange[1],yrange[0],yrange[1]], zorder=0 )


   # --- PRISTINE ---
   m = ptr['m'][:,i]
   x = r[:,i,0]  
   y = r[:,i,1]
   il = i-navg
   ir = i+navg
   if il < 0:
     il = 0
   vx = np.mean( v[:,il:ir,0], axis=1 )
   vy = np.mean( v[:,il:ir,1], axis=1 )
   vel = np.sqrt(vx**2 + vy**2)
   vx /= vel
   vy /= vel
   itr = (ptr['Ptype'][:,i] == 0) & (f['IsPristine'][:] == 1) & (f['IsWind'][:,i] == 0)
   mm_map, mvx_map, mvy_map = daa.make_2Dgrid( x[itr], y[itr], xrange, yrange=yrange, pixels=40, weight1=m[itr], weight2=m[itr]*vx[itr], weight3=m[itr]*vy[itr] )
   mm_map = daa.clip_2Dgrid( mm_map )
   vx_map = mvx_map / mm_map
   vy_map = mvy_map / mm_map
   Xgrid = np.linspace(xrange[0], xrange[1], mm_map.shape[0])  
   Ygrid = np.linspace(yrange[0], yrange[1], mm_map.shape[1])
   #ax.streamplot( Xgrid, Ygrid, vx_map.T, vy_map.T, color='plum', density=[1.,1.], linewidth=0.8, arrowsize=1.5 )
   ax.streamplot( Xgrid, Ygrid, vx_map.T, vy_map.T, color='violet', density=[1.,1.], linewidth=1., arrowsize=2.3 )


   # --- WIND RECYCLING ---
   #ind_gas_wind = np.where( (ptr['Ptype'][:,i] == 0) & (f['IsPristine'][:] == 1) & (f['IsWind'][:,i] == 1) & (f['Neject'][:] > 1) )[0]
   ind_gas_wind = np.where( (ptr['Ptype'][:,i] == 0) & (IsInGalID[:,i] == 1) & (np.sum(IsEjected[:,i-nstp:i],axis=1) >= 1) )[0]
   np.random.shuffle(ind_gas_wind)
   #for j in range(ind_gas_wind.size):
   for j in range( np.min([ind_gas_wind.size,ntest]) ):
      iacc = np.where( IsGasAccreted[ind_gas_wind[j],0:i] == 1)[0]
      if (iacc.size == 0):
        continue
      iacc = iacc[-1]
      iaft = i
      if iacc > iaft:
        print 'problemssss...1'
        continue
      if (iaft-iacc < 2*nsmooth):    # not enough snaps to smooth trajectory
        continue
      xtrs = daa.mysmooth(r[ind_gas_wind[j],iacc:iaft,0],nsmooth,sfac=2.)
      ytrs = daa.mysmooth(r[ind_gas_wind[j],iacc:iaft,1],nsmooth,sfac=2.)
      rtrs = np.sqrt(xtrs**2 + ytrs**2)
      ztrs = z[iacc:iaft]
      lwtrs = np.linspace(0.1, 1.5, xtrs.size)
      cltrs = np.linspace(0., 1, xtrs.size)
      daa.colorline( xtrs, ytrs, z=cltrs, linewidth=lwtrs, cmap=cm.Blues, norm=cl.Normalize(vmin=-0.3,vmax=1.) )


   # --- INTERGALACTIC TRANSFER ---
   ind_gas_trans = np.where( (ptr['Ptype'][:,i] == 0) & (f['IsMassTransfer'][:] == 1) & (IsInOtherGal[:,i] == 1) & (IsAfterOtherGal[:,i-nstp] == 1) )[0]
   np.random.shuffle(ind_gas_trans)
   #for j in range(ind_gas_trans.size):
   for j in range( np.min([ind_gas_trans.size,ntest]) ):
      iacc = np.where( IsGasAccreted[ind_gas_trans[j],0:i] == 1)[0]
      iaft = np.where( IsAfterOtherGal[ind_gas_trans[j],:] == 1)[0]
      if (iacc.size == 0) or (iaft.size == 0):
        continue
      iacc = iacc[-1]
      #iaft = iaft[-1]      ## this is the default!
      iaft = i
      if iacc > iaft:
        print 'problemssss...2'
        continue
      if (iaft-iacc < 2*nsmooth):    # not enough snaps to smooth trajectory
        continue
      xtrs = daa.mysmooth(r[ind_gas_trans[j],iacc:iaft,0],nsmooth,sfac=2.)
      ytrs = daa.mysmooth(r[ind_gas_trans[j],iacc:iaft,1],nsmooth,sfac=2.)
      rtrs = np.sqrt(xtrs**2 + ytrs**2)
      ztrs = z[iacc:iaft]
      lwtrs = np.linspace(0.1, 1.5, xtrs.size)
      cltrs = np.linspace(0., 1, xtrs.size)
      daa.colorline( xtrs, ytrs, z=cltrs, linewidth=lwtrs, cmap=cm.Greens, norm=cl.Normalize(vmin=-0.3,vmax=1.) )
   #ax.scatter( r[ind_gas_trans,i,0], r[ind_gas_trans,i,1], s=15, c='lightgreen', edgecolors='green', alpha=0.2, zorder=0 )  



   # --- plot star particle positions
   sz=30
   """
   ind_star = np.where( ptr['Ptype'][:,i] == 4 )[0]
   np.random.shuffle(ind_star)
   if ind_star.size > ntest:
     ax.scatter( r[ind_star[0:ntest],i,0], r[ind_star[0:ntest],i,1], s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )
   else:
     ax.scatter( r[ind_star,i,0], r[ind_star,i,1], s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )
   """
   P = g.readsnap( simdir, snapnum, 4, cosmological=1 )
   x = P['p'][:,0] - skidgal['p_phi'][i,0]
   y = P['p'][:,1] - skidgal['p_phi'][i,1]
   nstar = x.size
   if nstar > ntestS:
     ind_star = np.arange(nstar)
     np.random.shuffle( ind_star )
     ax.scatter( x[ind_star[0:ntestS]], y[ind_star[0:ntestS]], s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )
   else:
     ax.scatter( x, y, s=sz, c='yellow', edgecolors='orange', marker='*', alpha=0.5, zorder=2 )


   #--- plot SKID galaxies
   #for ng in xrange(Ngal-1):
   #  #if gal['npart'][ng] > 1000:
   #   xcen = gal['p_phi'][ng,0] - cen_pos[0]
   #   ycen = gal['p_phi'][ng,1] - cen_pos[1]
   #   #rcen = gal['Rout'][ng]
   #   rcen = 2. * gal['ReStar'][ng]
   #   ax.plot( xcen+rcen*xcirc, ycen+rcen*ycirc, '-', color='red', linewidth=lw)
   #   ax.plot( xcen, ycen, '+', color='red', ms=15, mew=mew )

   # --- plot Rvir for three most massive halos
   #halo = daa.read_AHF_halos(simdir + 'AHF', snapnum, readheader=False, h=P['hubble'] )   
   halo = daa.read_AHF_halos(simdir, snapnum, readheader=False, h=h )
   for j in range(halo['Rvir'].size):
      if halo['M_star'][j] < 1e6:
        continue
   #for j in range(10):
   #   if halo['HostID'][j] >= 0:
   #     continue
      xcen = halo['pos'][j,0] - skidgal['p_phi'][i,0]
      ycen = halo['pos'][j,1] - skidgal['p_phi'][i,1]
      zcen = halo['pos'][j,2] - skidgal['p_phi'][i,2]
      rcen = halo['Rvir'][j]
      if np.abs(zcen) <= 2.*xrange[1]:
        ax.plot( xcen+rcen*xcirc, ycen+rcen*ycirc, '--', color='black', linewidth=0.5, zorder=10 )

   """
   #--- plot MAIN SKID galaxy
   xcen = 0
   ycen = 0
   rcen = 1. * skidgal['ReStar'][i]
   ax.plot( xcen+rcen*xcirc, ycen+rcen*ycirc, '-', color='red', linewidth=lw)
   """

   print 'snapnum = %d   redshift = %.4f %.4f    ngas = %d   nstar = %d' % ( snapnum, redshift, skidgal['redshift'][i], ind_gas.size, ind_star.size )
   sys.stdout.flush()

   fig.savefig(outdir + figname, rasterized=rasterized, dpi=150) 
   plt.close()


'Done!'
