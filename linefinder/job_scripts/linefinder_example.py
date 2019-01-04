import linefinder.linefinder as linefinder

########################################################################

sim_name = 'm12i'
'''The simulation to run tracking on.'''

tag = '{}_example'.format( sim_name )
'''Identifying tag used as part of the filenames.
E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
'''

# Tracking Parameters
tracker_kwargs = {
    # What particle types to track. Typically just stars and gas.
    'p_types': [ 0, 4,],

    # What snapshots to compile the particle tracks for.
    'snum_start': 1,
    'snum_end': 600,
    'snum_step': 1,
}

# This is the actual function that runs linefinder.
# In general you don't need to touch this function but if you want to,
# for example, turn off one of the steps because you're rerunning and you
# already did that step, you can do so below.
linefinder.run_linefinder_jug(
    sim_name = sim_name,
    tag = tag,
    galdef = '_galdefv3',
    # The galdef is a set of parameters used for the galaxy linking and
    # classification steps. Don't touch this unless you know what you're doing.
    tracker_kwargs = tracker_kwargs,
    run_id_selecting = False,
    # run_id_sampling = False,
    # run_tracking = False,
    # run_galaxy_linking = False,
    # run_classifying = False,
)