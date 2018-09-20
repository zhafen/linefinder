#!/usr/bin/env python
'''Non-physical constants for use in analyzing data.
For example, this contains the default fill values for invalid data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import utils.presentation_constants as p_constants

########################################################################
# Global Default Parameters
########################################################################
# I do it this way that way I can access these global defaults from plotting
# scripts as well,
# and also have all the most important parameters shown in one place.

# The radius of the galaxy is defined as R_{gal} = GALAXY_CUT*LENGTH_SCALE
LENGTH_SCALE = 'Rstar0.5'
MT_LENGTH_SCALE = 'sRstar0.5'
GALAXY_CUT = 4.0

# The CGM is defined as max( (1. + F_GAP)*Rgal, INNER_CGM_BOUNDARY*Rvir )
#  to OUTER_CGM_BOUNDARY*Rvir.
INNER_CGM_BOUNDARY = 0.1
OUTER_CGM_BOUNDARY = 1.0
F_GAP = 0.2

# Sometimes we may also require the following density cut for material to be
# part of a galaxy (n_b in cm^-3)
GALAXY_DENSITY_CUT = 0.1

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

GALAXY_DEFINITIONS = {
    '' : {
        'length_scale' : 'Rstar0.5',
        'galaxy_cut' : 5.0,
    },
    '_galdefv1' : {
        'length_scale' : 'Rvir',
        'galaxy_cut' : 0.1,
    },
    '_galdefv2' : {
        'length_scale' : 'Rstar0.5',
        'galaxy_cut' : 5.0,

        't_pro' : 0.1,
        't_m' : 0.5,
    },
    '_galdefv3' : {
        'length_scale' : 'Rstar0.5',
        'mt_length_scale' : 'sRstar0.5',
        'galaxy_cut' : 4.0,

        't_pro' : 0.03,
        't_m' : 0.5,
    },
    '_galdefv4' : {
        'length_scale' : 'Rstar0.5',
        'mt_length_scale' : 'sRstar0.5',
        'galaxy_cut' : 5.0,

        't_pro' : 0.03,
        't_m' : 0.5,
    },
}

########################################################################
# System Information
########################################################################

ACTIVE_SYSTEM = 'Stampede2'

QUEST_PARAMETERS = {
    'simulation_data_dir': '/projects/b1026/zhafen',
    'linefinder_data_dir': '/projects/b1026/zhafen/linefinder_data',
    'halo_data_dir': '/projects/b1026/zhafen',
}

STAMPEDE_PARAMETERS = {
    'linefinder_data_dir': '/work/03057/zhafen/linefinder_data',
    'halo_data_dir': '/scratch/03057/zhafen',
}

STAMPEDE2_PARAMETERS = {
    'simulation_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE',
    'linefinder_data_dir': '/scratch/03057/zhafen/linefinder_data',
    'halo_data_dir': '/scratch/03057/zhafen',

    'project' : {
        'CGM_origin' : {
            'project_dir' : '/home1/03057/zhafen/papers/CGM_origin',
            'presentation_dir' : '/work/03057/zhafen/presentation_plots',
            'extras_dir' : '/work/03057/zhafen/extra_plots/CGM_origin',
        },
        'CGM_fate' : {
            'project_dir' : '/home1/03057/zhafen/papers/CGM_fate',
            'presentation_dir' : '/work/03057/zhafen/presentation_plots',
            'extras_dir' : '/work/03057/zhafen/extra_plots/CGM_fate',
        },
        'galaxy_origin' : {
            'project_dir' : '/home1/03057/zhafen/papers/galaxy_origin',
        },
    },
}

JUG_PATH = '~/.local/bin/jug'
JUG_EXEC_PATH = '~/.local/bin/jug-execute'

########################################################################
# Simulation Information
########################################################################

FULL_SIM_NAME = {
    'm10q': 'm10q_res250',
    'm10v': 'm10v_res250',
    'm10y': 'm10y_res250',
    'm10z': 'm10z_res250',
    'm11a': 'm11a_res2100',
    'm11b': 'm11b_res2100',
    'm11q': 'm11q_res7100',
    'm11v': 'm11v_res7100',
    'm11c': 'm11c_res2100',
    'm11d': 'm11d_res7100',
    'm11e': 'm11e_res7100',
    'm11h': 'm11h_res7100',
    'm11i': 'm11i_res7100',
    'm12b': 'm12b_res7100',
    'm12c': 'm12c_res7100',
    'm12f': 'm12f_res7100',
    'm12i': 'm12i_res7100',
    'm12m': 'm12m_res7100',
    'm12r': 'm12r_res7100',
    'm12w': 'm12w_res7100',
    'm12z': 'm12z_res4200',
}
MAIN_MT_HALO_ID = {
    'm10q': 0,
    'm10v': 2,
    'm10y': 0,
    'm10z': 0,
    'm11a': 0,
    'm11b': 0,
    'm11q': 0,
    'm11v': 0,
    'm11c': 0,
    'm12i': 0,
    'm12f': 0,
    'm12m': 0,
    'm11d_md': 0,
    'm11e_md': 0,
    'm11h_md': 0,
    'm11i_md': 0,
    'm10q_md': 0,
    'm11q_md': 0,
    'm12b_md': 0,
    'm12c_md': 0,
    'm12i_md': 0,
    'm12r_md': 0,
    'm12w_md': 0,
    'm12z_md': 0,
}

MASS_BINS = {
    'm10q': 'm10',
    'm10v': 'm10',
    'm10y': 'm10',
    'm10z': 'm10',
    'm11a': 'm10',
    'm11b': 'm10',
    'm11q': 'm11',
    'm11v': 'm11',
    'm11c': 'm11',
    'm12i': 'm12',
    'm12f': 'm12',
    'm12m': 'm12',
    'm10q_md': 'm10',
    'm11i_md': 'm10',
    'm11d_md': 'm11',
    'm11e_md': 'm11',
    'm11h_md': 'm11',
    'm11q_md': 'm11',
    'm12b_md': 'm12',
    'm12c_md': 'm12',
    'm12i_md': 'm12',
    'm12r_md': 'm12',
    'm12w_md': 'm12',
    'm12z_md': 'm12',
}

FULL_PHYSICS_NAME = {
    '': 'core',
    '_md': 'metal_diffusion',
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

########################################################################
# Presentation information
########################################################################

COLORSCHEME = p_constants.CLASSIFICATION_COLORS_B
