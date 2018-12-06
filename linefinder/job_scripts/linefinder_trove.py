import numpy as np
import sys

import linefinder.config as p_config
import linefinder.linefinder as linefinder
import linefinder.utils.file_management as file_management
import linefinder.utils.trove_management as trove_management

########################################################################

sim_names = [
    # 'm10q',
    # 'm10v',
    'm10y',
    # 'm10z',
    # 'm11q',
    # 'm11v',
    'm11a',
    # 'm11b',
    # 'm11c',
    'm12i',
    'm12f',
    # 'm12m',
    # 'm10q_md',
    # 'm11q_md',
    # 'm12i_md',
    # 'm12b_md',
    # 'm12c_md',
    # 'm12z_md',
    # 'm12r_md',
    # 'm12w_md',
]
galdefs = [
    # '',
    # '_galdefv1',
    '_galdefv2',
]

# Get the file format
ptracks_tag_format = '{}'
tag_format = '{}{}'.format( ptracks_tag_format, '{}' )
file_format =  'classifications_{}.hdf5'.format( tag_format )

# Start up a trove manager and use it to get next args
trove_manager = trove_management.LinefinderTroveManager(
    file_format,
    sim_names,
    galdefs,
)
args_to_use = trove_manager.get_next_args_to_use()

# DEBUG
# args_to_use = ( 'm12i', '_galdefv2' )

sim_name = args_to_use[0]
galdef = args_to_use[1]
ptracks_tag = ptracks_tag_format.format( *args_to_use[:-1] )
tag = tag_format.format( *args_to_use )

galdef_dict = p_config.GALAXY_DEFINITIONS[galdef]

print( "Running data {}".format( tag ) )

p_types = [ 0, 4, ]

selector_kwargs = {
    'tag' : ptracks_tag,

    'snum_start': 50,
    'snum_end': 600,
    'snum_step': 1,

    'p_types': p_types,

    'snapshot_kwargs': {
        'ahf_index': 600,
        'length_scale_used': 'Rstar0.5',
        'load_additional_ids' : False,
        'ahf_tag' : 'smooth',
    },

    # 'n_processors' : 3,
}

selector_data_filters = {
    # The negative is to avoid removing particles exactly at (0,0,0)
    'Radial Cut' : { 'data_key': 'Rf', 'data_min': -0.1, 'data_max': 5.0, },
    # We would include a density cut too, but because we're tracking both
    # stars and gas, this doesn't work yet.
    # 'Density Cut' : { 'data_key': 'HDen', 'data_min': 0.1, 'data_max': np.inf, }
}

sampler_kwargs = {
    'tag' : ptracks_tag,

    'ignore_duplicates': True,
}

# Tracking Parameters
tracker_kwargs = {
    'tag' : ptracks_tag,

    'p_types': p_types,

    'snum_start': 1,
    'snum_end': 600,
    'snum_step': 1,
}

# Galaxy Finding Parameters
gal_linker_kwargs = {
    'ptracks_tag' : ptracks_tag,
    'galaxy_cut' : galdef_dict['galaxy_cut'],
    'length_scale' : galdef_dict['length_scale'],
}

# Classifying Parameters
classifier_kwargs = {
    'ptracks_tag' : ptracks_tag,
}

linefinder.run_linefinder(
    sim_name = sim_name,
    tag = tag,
    selector_data_filters = selector_data_filters,
    selector_kwargs = selector_kwargs,
    sampler_kwargs = sampler_kwargs,
    tracker_kwargs = tracker_kwargs,
    gal_linker_kwargs = gal_linker_kwargs,
    classifier_kwargs = classifier_kwargs,
    # run_id_selecting = False,
    # run_id_sampling = False,
    # run_tracking = False,
    # run_galaxy_linking = False,
    # run_classifying = False,
)
