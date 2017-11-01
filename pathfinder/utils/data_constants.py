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
# Global Defaults
########################################################################

# Fiducial threshold processing time, t_{pro}
T_PRO = 100.
