import numpy as np
import sys

import linefinder.config as config
import linefinder.linefinder as linefinder
import linefinder.utils.file_management as file_management
import linefinder.utils.trove_management as trove_management

########################################################################

sim_name = 'm12i'
'''The simulation to run tracking on.'''

# Identifying tag used as part of the filenames.
tag = 'example'
'''Identifying tag used as part of the filenames.'''

# We need to choose what our definition of a galaxy is.
# The standard choice is _galdefv3, which defines the galaxy radius as
# 4 R_half. The choice of galaxy definition may also affect your
# classifications, e.g. the amount of time a particle needs to stay in a
# galaxy to count as "preprocessed".
# If you don't know what to set this to, leave it.
galdef = '_galdefv3'
galdef_dict = config.GALAXY_DEFINITIONS[galdef]

p_types = [ 0, 4, ]

# Tracking Parameters
tracker_kwargs = {

    'p_types': [ 0, 4,],

    'snum_start': 1,
    'snum_end': 600,
    'snum_step': 1,
}

# Galaxy Finding Parameters
gal_linker_kwargs = {
    'galaxy_cut' : galdef_dict['galaxy_cut'],
    'length_scale' : galdef_dict['length_scale'],
    'mt_length_scale' : galdef_dict['mt_length_scale'],
}

# Classifying Parameters
classifier_kwargs = {
    't_pro': galdef_dict['t_pro'],
    't_m': galdef_dict['t_m'],
}

linefinder.run_linefinder_jug(
    sim_name = sim_name,
    tag = tag,
    tracker_kwargs = tracker_kwargs,
    gal_linker_kwargs = gal_linker_kwargs,
    classifier_kwargs = classifier_kwargs,
    run_id_selecting = False,
    # run_id_sampling = False,
    # run_tracking = False,
    # run_galaxy_linking = False,
    # run_classifying = False,
)
