import linefinder.galaxy_link as galaxy_link

parallel_kwargs = {
    'length_scale': 'Rvir',
    'mt_length_scale': 'Rvir',
    'ids_to_return': [
        'gal_id',
        'host_gal_id',
        'mt_gal_id',
        'd_gal',
        'd_other_gal_scaled',
    ],
    'minimum_criteria': 'n_star',
    'minimum_value': 0,

    'galaxy_cut': 0.1,

    'halo_data_dir': './linefinder/tests/data/ahf_test_data',
    'out_dir': './linefinder/tests/data/tracking_output',
    'ptracks_tag': 'test',
    'tag': 'test_jug',
    'mtree_halos_index': 600,
    'main_mt_halo_id': 0,
    'n_processors': 2,
}

particle_track_gal_linker = galaxy_link.ParticleTrackGalaxyLinker(
    **parallel_kwargs )
particle_track_gal_linker.find_galaxies_for_particle_tracks_jug()
