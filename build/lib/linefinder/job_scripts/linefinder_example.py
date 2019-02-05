import linefinder.linefinder as linefinder

########################################################################

linefinder_args = {
    # Identifying tag used as part of the filenames.
    # E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
    'tag': 'm12i_example',
    
    # Location to place output in
    'out_dir': '$SCRATCH/linefinder_data/core/m12i_res7100/data',

    # Location of simulation data
    'sim_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/core/m12i_res7100/output',

    # Location of halo file data
    'halo_data_dir': '$SCRATCH/core/m12i_res7100/halo',

    # Arguments used for the particle tracking step
    'tracker_kwargs': {
        # What particle types to track. Typically just stars and gas.
        'p_types': [ 0, 4,],

        # What snapshots to compile the particle tracks for.
        'snum_start': 1,
        'snum_end': 600,
        'snum_step': 1,
    },

    # Arguments used for the visualization step
    'visualization_kwargs': {
        # These kwargs are used for tuning the Firefly visualization
        'export_to_firefly_kwargs': {
            'firefly_dir': '$SCRATCH/firefly_repos/firefly_trails',
            'firefly_source': 'git@github.com:zhafen/Firefly.git',
        },
    },

    # The following arguments are for turning on/off different parts
    # of the pipeline
    'run_id_selecting': False,
    # Most users will identify the list of IDs using their own methods, so
    # we turn ID selecting off.
    'run_id_sampling': True,
    'run_tracking': True,
    'run_galaxy_linking': True,
    'run_classifying': True,
    'run_visualization': True,
}

# Actually run Linefinder
linefinder.run_linefinder_jug(
    **linefinder_args
)
