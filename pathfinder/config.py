#!/usr/bin/env python
'''Non-physical constants for use in analyzing data.
For example, this contains the default fill values for invalid data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

########################################################################
# Global Default Parameters
########################################################################
# I do it this way that way I can access these global defaults from plotting
# scripts as well,
# and also have all the most important parameters shown in one place.

# The radius of the galaxy is defined as R_{gal} = GALAXY_CUT*LENGTH_SCALE
LENGTH_SCALE = 'Rvir'
GALAXY_CUT = 0.1

# The CGM is defined as INNER_CGM_BOUNDARY*Rvir to OUTER_CGM_BOUNDARY*Rvir
INNER_CGM_BOUNDARY = 0.1
OUTER_CGM_BOUNDARY = 1.0

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

# Fiducial threshold processing time, t_{pro}, in Gyr.
T_PRO = 0.1

# Fiducial time interval during which material can be counted as merger (in Gyr)
T_M = 0.5

########################################################################
# System Information
########################################################################

ACTIVE_SYSTEM = 'Stampede2'

QUEST_PARAMETERS = {
    'simulation_data_dir': '/projects/b1026/zhafen',
    'pathfinder_data_dir': '/projects/b1026/zhafen/pathfinder_data',
    'ahf_data_dir': '/projects/b1026/zhafen',
}

STAMPEDE_PARAMETERS = {
    'pathfinder_data_dir': '/work/03057/zhafen/pathfinder_data',
    'ahf_data_dir': '/scratch/03057/zhafen',
}

STAMPEDE2_PARAMETERS = {
    'simulation_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/core',
    'pathfinder_data_dir': '/scratch/03057/zhafen/pathfinder_data',
    'ahf_data_dir': '/scratch/03057/zhafen',
}

JUG_PATH = '~/.local/bin/jug'
JUG_EXEC_PATH = '~/.local/bin/jug-execute'

########################################################################
# Simulation Information
########################################################################

FULL_SIM_NAME = {
    'm10q': 'm10q_res250',
    'm10v': 'm10v_res250',
    'm11q': 'm11q_res7100',
    'm11v': 'm11v_res7100',
    'm12i': 'm12i_res7100',
    'm12f': 'm12f_res7100',
    'm12m': 'm12m_res7100',
    'm12imd': 'm12i_res7100_md',
}

########################################################################
# Miscellanious values
########################################################################

# What fill values to use throughout?
INT_FILL_VALUE = -99999
FLOAT_FILL_VALUE = np.nan

# What integer particle type does each component correspond to?
PTYPE_GAS = 0
PTYPE_DM = 1
PTYPE_LOWRES_DM = 2
PTYPE_STAR = 4
