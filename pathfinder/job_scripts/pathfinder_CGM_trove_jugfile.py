import sys

import pathfinder.config as p_config
import pathfinder.pathfinder as pathfinder
import pathfinder.utils.file_management as file_management
import pathfinder.utils.trove_management as trove_management

########################################################################

sim_names = [
    'm10q',
    'm10v',
    'm10y',
    'm10z',
    'm11q',
    'm11v',
    'm11a',
    'm11b',
    'm11c',
    'm12i',
    'm12f',
    'm12m',
    # 'm10q_md',
    # 'm11q_md',
    # 'm12i_md',
    'm12b_md',
    # 'm12c_md',
    'm12z_md',
    'm12r_md',
    'm12w_md',
]
snums = [
    # 600,
    # 578,
    # 556,
    # 534,
    # 513,
    # 492,
    # 486,
    # 471,
    465, # z = 0.25
    # 451,
    # 431,
    # 412,
    # 392,
    # 382, # z = 0.5
    # 373,
    # 354,
    # 335,
    # 316,
    # 297,
    # 277, # z = 1
    # 242,
    # 214,
    172, # z = 2
    # 156,
    # 142,
    # 120,
    # 88,
    # 67,
    # 52,
    # 41,
    # 33,
    # 20,
    # 0,
]
galdefs = [
    # '',
    '_galdefv1',
    '_galdefv2',
]

# Get the file format
ptracks_tag_format = '{}_CGM_snum{}'
tag_format = '{}{}'.format( ptracks_tag_format, '{}' )
file_format =  'ptracks_{}.hdf5'.format( ptracks_tag_format )

# Start up a trove manager and use it to get next args
trove_manager = trove_management.PathfinderTroveManager(
    file_format,
    sim_names,
    snums,
    galdefs,
)
args_to_use = trove_manager.get_next_args_to_use()

sim_name = args_to_use[0]
snum = args_to_use[1]
galdef = args_to_use[2]
ptracks_tag = ptracks_tag_format.format( *args_to_use[-1] )
tag = tag_format.format( *args_to_use )

galdef_dict = p_config.GALAXY_DEFINITIONS[galdef]

print( "Running data {}".format( tag ) )

p_types = [ 0, 4, ]

selector_kwargs = {
    'snum_start': snum,
    'snum_end': snum,
    'snum_step': 1,

    'p_types': p_types,

    'snapshot_kwargs': {
        'ahf_index': 600,
        'length_scale_used': 'R_vir',
    }
}

selector_data_filters = {
  'CGM' : { 'data_key': 'Rf', 'data_min': 0.1, 'data_max': 1.0, }
}

sampler_kwargs = {
    'ignore_duplicates': True,
}

# Tracking Parameters
tracker_kwargs = {
    'p_types': p_types,

    'snum_start': 1,
    'snum_end': 600,
    'snum_step': 1,
}

# Galaxy Finding Parameters
gal_finder_kwargs = {
    'ptracks_tag' : ptracks_tag,
    'galaxy_cut' : galdef_dict['galaxy_cut'],
    'length_scale' : galdef_dict['length_scale'],
}

# Classifying Parameters
classifier_kwargs = {
    'ptracks_tag' : ptracks_tag,
}

pathfinder.run_pathfinder_jug(
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
