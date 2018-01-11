#!/usr/bin/env python
'''Non-physical constants for use in analyzing data.
For example, this contains the default fill values for invalid data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

########################################################################
########################################################################

JUG_PATH = '~/.local/bin/jug'
JUG_EXEC_PATH = '~/.local/bin/jug-execute'

########################################################################
########################################################################

INT_FILL_VALUE = -99999
FLOAT_FILL_VALUE = np.nan

PTYPE_GAS = 0
PTYPE_DM = 1
PTYPE_LOWRES_DM = 2
PTYPE_STAR = 4

########################################################################
# Global Default Parameters
########################################################################
# I do it this way that way I can access these global defaults from plotting
# scripts as well,
# and also have all the most important parameters shown in one place.

# The radius of the galaxy is defined as R_{gal} = GALAXY_CUT*LENGTH_SCALE
LENGTH_SCALE = 'Rstar0.5'
GALAXY_CUT = 5.0

# Sometimes we may also require the following density cut for material to be
# part of a galaxy (n_b in cm^-3)
GALAXY_DENSITY_CUT = 0.075  # I.E. nH = 0.1 cm^-3

# The ejection velocity necessary to be counted as a wind is defined as
# v_{wind} = WIND_CUT*VELOCITY_SCALE
# There's also an absolute wind velocity in km/s required,
# v_{wind} = ABSOLUTE_WIND_CUT
VELOCITY_SCALE = 'Vmax'
WIND_CUT = 1.0
ABSOLUTE_WIND_CUT = 15.0

# Fiducial threshold processing time, t_{pro}, in Myr.
T_PRO = 100.0

# Fiducial time interval during which material can be counted as merger (in Myr)
T_M = 500.0