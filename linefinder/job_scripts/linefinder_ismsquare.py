import linefinder.linefinder as linefinder

########################################################################

linefinder_args = {
    # Identifying tag used as part of the filenames.
    # E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
    'tag': 'm12i_ismsquare2',
    
    # Location to place output in
    'out_dir': '$SCRATCH/linefinder_data/core/m12i_res7100/data',

    # Location of simulation data
    'sim_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/core/m12i_res7100/output',

    # Location of halo file data
    'halo_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/core/m12i_res7100/halo/ahf',

    # Arguments for id sampling
    'sampler_kwargs': {
        'ignore_duplicates': True,
        'n_samples': 200000,
        'p_types': [ 0, 4,],
        'snapshot_kwargs': {
            'ahf_index': 600,
            'length_scale_used': 'R_vir',
        },
    },

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
        'install_firefly': False,
        # 'write_startup': True,
        # These kwargs are used for tuning the Firefly visualization
        'export_to_firefly_kwargs': {
            'firefly_dir': '/work/03057/zhafen/firefly_repos/ismsquare',
            'snum': 550,
            'pathline_inds_to_display': range(48,51),
            'n_pathlines': 100000,
            'classifications': [ None, 'will_leaves_gal_dt_0.050', 'is_cluster_star', ],
            'classification_ui_labels': [ 'All', 'EjectedSoon', 'Clusters', ],
            # 'use_default_colors': False,
            'size_mult': 1,
        },
    },

    # The following arguments are for turning on/off different parts
    # of the pipeline
    'run_id_selecting': False,
    # Most users will identify the list of IDs using their own methods, so
    # we turn ID selecting off.
    'run_id_sampling': False,
    'run_tracking': False,
    'run_galaxy_linking': False,
    'run_classifying': False,
    'run_visualization': True,
}

# Actually run Linefinder
linefinder.run_linefinder_jug(
    **linefinder_args
)
