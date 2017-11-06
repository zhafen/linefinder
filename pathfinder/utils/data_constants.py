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

INT_FILL_VALUE = -99999
FLOAT_FILL_VALUE = np.nan

PTYPE_GAS = 0
PTYPE_DM = 1
PTYPE_LOWRES_DM = 2
PTYPE_STAR = 4

########################################################################
# Global Default Parameters
########################################################################
# I do it this way that way I can access these global defaults from plotting scripts as well,
# and also have all the most important parameters shown in one place.

# Fiducial threshold processing time, t_{pro}
T_PRO = 50.0

# Fiducial time interval during which material can be counted as merger.
T_M = 500.0
