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
    # 'm10y',
    # 'm10z',
    # 'm11q',
    # 'm11v',
    # 'm11a',
    # 'm11b',
    # 'm11c',
    'm12i',
    # 'm12f',
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
c_tag_format = '{}_tm{}'.format( tag_format, sys.argv[1] )
file_format =  'classifications_{}.hdf5'.format( c_tag_format )

# Start up a trove manager and use it to get next args
trove_manager = trove_management.LinefinderTroveManager(
    file_format,
    sim_names,
    galdefs,
)
args_to_use = trove_manager.get_next_args_to_use()

sim_name = args_to_use[0]
galdef = args_to_use[1]
ptracks_tag = ptracks_tag_format.format( *args_to_use[:-1] )
tag = tag_format.format( *args_to_use )
classifications_tag = c_tag_format.format( *args_to_use )

galdef_dict = p_config.GALAXY_DEFINITIONS[galdef]

print( "Running data {}".format( tag ) )

# Classifying Parameters
classifier_kwargs = {
    'ptracks_tag' : ptracks_tag,
    'galids_tag' : tag,
    'tag' : classifications_tag,
    't_m' : float( sys.argv[1] ),
}

linefinder.run_linefinder(
    sim_name = sim_name,
    tag = tag,
    classifier_kwargs = classifier_kwargs,
    run_id_selecting = False,
    run_id_sampling = False,
    run_tracking = False,
    run_galaxy_linking = False,
    # run_classifying = False,
)
