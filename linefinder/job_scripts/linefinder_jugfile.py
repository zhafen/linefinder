import numpy as np
import sys

import linefinder.config as p_config
import linefinder.linefinder as linefinder
import linefinder.utils.file_management as file_management
import linefinder.utils.trove_management as trove_management

########################################################################

# sim_names = [
#     # 'm10q',
#     # 'm10v',
#     # 'm10y',
#     # 'm10z',
#     # 'm11q',
#     # 'm11v',
#     # 'm11a',
#     # 'm11b',
#     # 'm11c',
#     'm12i',
#     # 'm12f',
#     # 'm12m',
#     # 'm10q_md',
#     # 'm11q_md',
#     # 'm12i_md',
#     # 'm12b_md',
#     # 'm12c_md',
#     # 'm12z_md',
#     # 'm12r_md',
#     # 'm12w_md',
# ]
# 
# # Get the file format
# ptracks_tag_format = '{}'
# tag_format = '{}{}'.format( ptracks_tag_format, '{}' )
# file_format =  'galids_{}.hdf5'.format( tag_format )
# 
# # Start up a trove manager and use it to get next args
# trove_manager = trove_management.LinefinderTroveManager(
#     file_format,
#     sim_names,
#     galdefs,
# )
# args_to_use = trove_manager.get_next_args_to_use()
# 
# sim_name = args_to_use[0]
# ptracks_tag = ptracks_tag_format.format( *args_to_use[:-1] )
# tag = tag_format.format( *args_to_use )

# We need to choose what our definition of a galaxy is.
# The standard choice is _galdefv3, which defines the galaxy radius as
# 4 R_half. The choice of galaxy definition may also affect your
# classifications, e.g. the amount of time a particle needs to stay in a
# galaxy to count as "preprocessed".
# If you don't know what to set this to, leave it.
galdef = '_galdefv3'
galdef_dict = p_config.GALAXY_DEFINITIONS[galdef]

p_types = [ 0, 4, ]

# selector_kwargs = {
# 
#     'snum_start': 50,
#     'snum_end': 600,
#     'snum_step': 1,
# 
#     'p_types': p_types,
# 
#     'snapshot_kwargs': {
#         'ahf_index': 600,
#         'length_scale_used': 'Rstar0.5',
#         'load_additional_ids' : False,
#         'ahf_tag' : 'smooth',
#     },
# 
#     # 'n_processors' : 3,
# }

# selector_data_filters = {}

sampler_kwargs = {
    'ignore_duplicates': True,
}

# Tracking Parameters
tracker_kwargs = {

    'p_types': [ 0, 4,],

    'snum_start': 1,
    'snum_end': 600,
    'snum_step': 1,
}

# Galaxy Finding Parameters
gal_finder_kwargs = {
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
    selector_data_filters = selector_data_filters,
    selector_kwargs = selector_kwargs,
    sampler_kwargs = sampler_kwargs,
    tracker_kwargs = tracker_kwargs,
    gal_finder_kwargs = gal_finder_kwargs,
    classifier_kwargs = classifier_kwargs,
    # run_id_selecting = False,
    # run_id_sampling = False,
    # run_tracking = False,
    # run_galaxy_finding = False,
    # run_classifying = False,
)
