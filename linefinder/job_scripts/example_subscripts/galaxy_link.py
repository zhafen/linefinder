'''Link particles to galaxies.'''

import sys

import linefinder.galaxy_link as galaxy_link

import trove

########################################################################

pm = trove.link_params_to_config(
    config_fp = sys.argv[1],
)

gal_linker = galaxy_link.ParticleTrackGalaxyLinker(
    out_dir = pm['data_dir'],
    tag = pm['tag'],
    main_mt_halo_id = 0,
    halo_data_dir = pm['halo_data_dir'],
    # This is only okay to use for low-z-focused analysis!
    mt_length_scale = 'Rstar0.5',
)
gal_linker.find_galaxies_for_particle_tracks_jug()
    

