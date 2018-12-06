import sys
import time

import jug

import linefinder.config as config
import linefinder.linefinder as linefinder
import linefinder.utils.file_management as file_management
import linefinder.utils.trove_management as trove_management

########################################################################

sim_names = [
    # 'm10q',
    'm11q',
    # 'm12i',
]
snums = [
    600,
    578,
    556,
    534,
    513,
    492,
    486,
    471,
    465,
    451,
    431,
    412,
    392,
    382,
    373,
    354,
    335,
    316,
    297,
    277,
    242,
    214,
    172,
    156,
    142,
    120,
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
file_format =  'classifications_{}.hdf5'.format( tag_format )

# Start up a trove manager and use it to get next args
trove_manager = trove_management.LinefinderTroveManager(
    file_format,
    sim_names,
    snums,
    galdefs,
)
args_to_use = trove_manager.get_next_args_to_use()

sim_name = args_to_use[0]
snum = args_to_use[1]
galdef = config.GALAXY_DEFINITIONS[args_to_use[2]]
ptracks_tag = ptracks_tag_format.format( *args_to_use[:-1] )
tag = tag_format.format( *args_to_use )

print( "Running {}".format( tag ) )

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
gal_linker_kwargs = {
    'length_scale' : galdef['length_scale'],
    'galaxy_cut' : galdef['galaxy_cut'],

    'ptracks_tag' : ptracks_tag,
}

# Classifying Parameters
classifier_kwargs = {
    'ptracks_tag' : ptracks_tag,
}

linefinder.run_linefinder_jug(
    sim_name = sim_name,
    tag = tag,
    selector_data_filters = selector_data_filters,
    selector_kwargs = selector_kwargs,
    sampler_kwargs = sampler_kwargs,
    tracker_kwargs = tracker_kwargs,
    gal_linker_kwargs = gal_linker_kwargs,
    classifier_kwargs = classifier_kwargs,
    run_id_selecting = False,
    run_id_sampling = False,
    run_tracking = False,
    # run_galaxy_linking = False,
    # run_classifying = False,
)

# Try adding this section so that the pipeline keeps going
# (If jug thinks it's finished, the pipeline breaks)
jug.barrier()
def dummy_fn():

    time.sleep( 10 )
jug.Task( dummy_fn )
