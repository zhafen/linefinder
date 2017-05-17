#!/usr/bin/env python
'''Tools for astronomical data processing.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

########################################################################

def hubble_z(redshift, h=0.702, Omega0=0.272, OmegaLambda=0.728):
  '''Return Hubble factor in 1/sec for a given redshift.

  Args:
    redshift (float): The input redshift.
    h (float): The hubble parameter.
    Omega0 (float): TODO
    OmegaLambda (float): TODO

  Returns:
    hubble_a (float): Hubble factor in 1/sec
  '''

   ascale = 1. / ( 1. + redshift )
   hubble_a = HUBBLE * h * np.sqrt( Omega0 / ascale**3 + (1. - Omega0 - OmegaLambda) / ascale**2 + OmegaLambda )     # in 1/sec !!

   return hubble_a

########################################################################


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
