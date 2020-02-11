import linefinder.linefinder as linefinder

########################################################################

linefinder_args = {
    # Identifying tag used as part of the filenames.
    # E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
    'tag': 'm12b_md_CGM',

    # Location to place output in
    'out_dir': '/scratch/03057/zhafen/linefinder_data/metal_diffusion/m12b_res7100/data',

    # Location of simulation data
    'sim_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12b_res7100/output',

    # Location of halo file data
    'halo_data_dir': '/scratch/03057/zhafen/metal_diffusion/m12b_res7100/halo',

    # ID Selection
    'selector_kwargs': {
        'snum_start': 55,
        'snum_end': 600,
        'snum_step': 1,

        'p_types': [ 0, 4, ],

        'snapshot_kwargs': {
            'ahf_index': 600,
            'length_scale_used': 'Rvir',
            'load_additional_ids' : False,
            'ahf_tag' : 'smooth',
        },

    },
    'selector_data_filters': {
        # The negative is to avoid removing particles exactly at (0,0,0)
        'Radial Cut' : { 'data_key': 'Rf', 'data_min': 0.1, 'data_max': 1.0, },
        # We would include a density cut too, but because we're tracking both
        # stars and gas, this doesn't work yet.
        # 'Density Cut' : { 'data_key': 'HDen', 'data_min': 0.1, 'data_max': np.inf, }
    },

    # Arguments for id sampling
    'sampler_kwargs': {
        'ignore_duplicates': True,
        'reference_snum_for_duplicates': 600,
    },

    # Arguments used for the particle tracking step
    'tracker_kwargs': {
        # What particle types to track. Typically just stars and gas.
        'p_types': [ 0, 4, ],

        # What snapshots to compile the particle tracks for.
        'snum_start': 1,
        'snum_end': 600,
        'snum_step': 1,
    },

    # Arguments used for galaxy linking
    'gal_linker_kwargs': {
        'galaxy_cut': 0.1,
        'length_scale': 'Rvir',
        'mt_length_scale': 'Rvir',
    },

    # The following arguments are for turning on/off different parts
    # of the pipeline
    'run_id_selecting': False,
    # Most users will identify the list of IDs using their own methods, so
    # we turn ID selecting off.
    'run_id_sampling': False,
    'run_tracking': True,
    'run_galaxy_linking': True,
    'run_classifying': True,
    # 'run_visualization': False,
}

# Actually run Linefinder
linefinder.run_linefinder_jug(
    **linefinder_args
)
