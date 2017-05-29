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

########################################################################
# Input Parameterss
########################################################################

# Number of tracked particles
#ntrack = 1e5                
ntrack = targeted_ids.size

# Simulation name
sname = 'm12i_res7000_md'

# What directories
sdir = '/scratch/projects/xsede/GalaxiesOnFIRE/metaldiff/{}/output'.format( sname )
outdir = '/scratch/03057/zhafen/{}/output'.format( sname )

targeted_id_filename = ''

# What particle types
Ptype = [ 0, 4 ]                     # must contain all possible particle types in idlist

# Snapshot range
snap_ini = 0
snap_end = 600                       # z = 0
snap_step = 1







