#!/usr/bin/env python
'''Constants used in the process of particle tracking.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

#########################################################################
# CONSTANTS
########################################################################

CM_PER_KM = 1e5
CM_PER_MPC = 3.085678e24
CM_PER_KPC = CM_PER_MPC / 1000.
HUBBLE = 3.2407789e-18             # in h/sec
PROTONMASS = 1.6726e-24            # in g
MSUN = 1.989e33                    # in g
SEC_PER_YEAR =  3.15569e7
G_UNIV = 6.672e-8                       # [ cm^3 g^-1 s^-2 ]
SPEED_OF_LIGHT = 2.99792458e10        # speed of light (cm s^-1)
SIGMA_T = 6.6524e-25                   # cm^(-2) Thomson cross-section
GAMMA = 5.0/3.0
GAMMA_MINUS1 = GAMMA - 1.

########################################################################
# CODE UNIT CONSTANTS
########################################################################
UNITMASS_IN_G            = 1.989e43        # 1.e10/h solar masses
#UnitMass_in_Msun         = UNITMASS_IN_G / MSUN  
UNITVELOCITY_IN_CM_PER_S = 1e5             # 1 km/s
UNITLENGTH_IN_CM         = 3.085678e21     # 1 kpc/h
#UnitLength_in_Mpc        = 0.001           
UNITTIME_IN_S            = UNITLENGTH_IN_CM / UNITVELOCITY_IN_CM_PER_S          # seconds / h
#UnitTime_in_Gyr          = UNITTIME_IN_S / ( SEC_PER_YEAR * 1e9 )
#UnitTime_in_Myr          = UNITTIME_IN_S / ( SEC_PER_YEAR * 1e6 )
#UnitDensity_in_cgs       = UNITMASS_IN_G / UNITLENGTH_IN_CM**3.
#UnitEnergy_in_cgs        = UNITMASS_IN_G * UNITLENGTH_IN_CM**2. / UNITTIME_IN_S**2.
#UnitMdot_in_Msun_per_yr = UnitMass_in_Msun / (UNITTIME_IN_S / SEC_PER_YEAR)
UNITG_UNIV = G_UNIV / UNITLENGTH_IN_CM**3 * UNITMASS_IN_G * UNITTIME_IN_S**2
#SolarAbundance           = 0.02
#
## ------------------------------ COSMOLOGY
#h = 0.6774   #0.702
#hinv = 1. / h
#rho_crit = 3 * (HUBBLE*h)**2 / ( 8 * np.pi * G_UNIV )       # in cgs
#Omega0 = 0.3089  #0.272
#OmegaLambda = 0.6911  #0.728
#OmegaBaryon = 0.0486  #0.0455