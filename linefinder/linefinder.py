#!/usr/bin/env python
'''This module contains functions for easily running all aspects of linefinder.
'''

import jug
import os

import config as linefinder_config
import select
import track
import galaxy_link
import classify

import utils.file_management as file_management

import galaxy_dive.utils.utilities as utilities

########################################################################
########################################################################


def run_linefinder(
    tag,
    out_dir = None,
    sim_name = None,
    galdef = None,
    selector_data_filters = {},
    selector_kwargs = {},
    sampler_kwargs = {},
    tracker_kwargs = {},
    gal_linker_kwargs = {},
    classifier_kwargs = {},
    run_id_selecting = True,
    run_id_sampling = True,
    run_tracking = True,
    run_galaxy_linking = True,
    run_classifying = True,
):
    '''Main function for running linefinder.

    Args:
        tag (str):
            Filename identifier for data products.

        out_dir (str):
            Output directory to store the data in.

        galdef (str):
            Which set of parameters to use for the galaxy_linking and
            classification steps?

        sim_name (str):
            Name of simulation to run linefinder for. If provided, linefinder
            will automatically fill in many arguments using a file_manager
            and the linefinder.config file.

        selector_data_filters (dict):
            Data filters to pass to select.IDSelector.select_ids()

        selector_kwargs (dict):
            Arguments to use when selecting what particles to track.
            Arguments will be passed to select.IDSelector

        sampler_kwargs (dict):
            Arguments to use when selecting what particles to track.
            Arguments will be passed to select.IDSampler

        tracker_kwargs (dict):
            Arguments to use when tracking particles.
            Arguments will be passedts to pass to track.ParticleTracker

        gal_linker_kwargs (dict):
            Arguments to use when associating particles with galaxies.
            Arguments will be passed to galaxy_link.ParticleTrackGalaxyLinker

        classifier_kwargs (dict):
            Arguments to use when classifying particles.
            Arguments will be passed to classify.Classifier

        run_id_selecting (bool):
            If True, then run routines for selecting particles.

        run_id_sampling (bool):
            If True, then run routines for sampling from the full list of
            selected particles.

        run_tracking (bool):
            If True, then run routines for tracking particles.

        run_galaxy_linking (bool):
            If True, then run routines for associating particles with galaxies.

        run_classifying (bool):
            If True, then run routines for classifying particles.
    '''

    if sim_name is not None:

        file_manager = file_management.FileManager()

        if out_dir is None:
            out_dir = file_manager.get_linefinder_dir( sim_name )

    if galdef is not None:
        galdef_dict = linefinder_config.GALAXY_DEFINITIONS[galdef]

    # These are kwargs that could be used at any stage of running linefinder.
    general_kwargs = {
        'out_dir': out_dir,
        'tag': tag,
    }

    # Run the ID Selecting
    if run_id_selecting:

        # Update arguments
        selector_kwargs = utilities.merge_two_dicts(
            selector_kwargs, general_kwargs )

        if sim_name is not None:
            snapshot_kwargs = selector_kwargs['snapshot_kwargs']

            if 'sdir' not in snapshot_kwargs:
                snapshot_kwargs['sdir'] = file_manager.get_sim_dir( sim_name )

            if 'halo_data_dir' not in snapshot_kwargs:
                snapshot_kwargs['halo_data_dir'] = file_manager.get_halo_dir( sim_name )

            if 'main_halo_id' not in snapshot_kwargs:
                snapshot_kwargs['main_halo_id'] = linefinder_config.MAIN_MT_HALO_ID[sim_name]

        id_selector = select.IDSelector( **selector_kwargs )
        id_selector.select_ids( selector_data_filters )

    # Run the ID Sampling
    if run_id_sampling:

        # Update arguments
        sampler_kwargs = utilities.merge_two_dicts(
            sampler_kwargs, general_kwargs )

        id_sampler = select.IDSampler( **sampler_kwargs )
        id_sampler.sample_ids()

    # Run the Particle Tracking
    if run_tracking:

        # Update arguments
        tracker_kwargs = utilities.merge_two_dicts(
            tracker_kwargs, general_kwargs )

        if 'sdir' not in tracker_kwargs:
            tracker_kwargs['sdir'] = file_manager.get_sim_dir( sim_name )

        particle_tracker = track.ParticleTracker( **tracker_kwargs )
        particle_tracker.save_particle_tracks()

    # Run the Galaxy Finding
    if run_galaxy_linking:

        # Update arguments
        gal_linker_kwargs = utilities.merge_two_dicts(
            gal_linker_kwargs, general_kwargs )

        if sim_name is not None:
            if 'halo_data_dir' not in gal_linker_kwargs:
                gal_linker_kwargs['halo_data_dir'] = file_manager.get_halo_dir( sim_name )
            if 'main_mt_halo_id' not in gal_linker_kwargs:
                gal_linker_kwargs['main_mt_halo_id'] = linefinder_config.MAIN_MT_HALO_ID[sim_name]

        if galdef is not None:
            for key in [ 'galaxy_cut', 'length_scale', 'mt_length_scale' ]:
                gal_linker_kwargs[key] = galdef_dict[key]

        particle_track_gal_linker = galaxy_link.ParticleTrackGalaxyLinker(
            **gal_linker_kwargs )
        particle_track_gal_linker.find_galaxies_for_particle_tracks()

    # Run the Classification
    if run_classifying:

        # Update arguments
        classifier_kwargs = utilities.merge_two_dicts(
            classifier_kwargs, general_kwargs )

        classifier = classify.Classifier( **classifier_kwargs )
        classifier.classify_particles()

########################################################################

def run_linefinder_jug(
    tag,
    out_dir = None,
    sim_name = None,
    galdef = None,
    selector_data_filters = {},
    selector_kwargs = {},
    sampler_kwargs = {},
    tracker_kwargs = {},
    gal_linker_kwargs = {},
    classifier_kwargs = {},
    run_id_selecting = True,
    run_id_sampling = True,
    run_tracking = True,
    run_galaxy_linking = True,
    run_classifying = True,
):
    '''Main function for running linefinder.

    Args:
        out_dir (str):
            Output directory to store the data in.

        tag (str):
            Filename identifier for data products.

        galdef (str):
            Which set of parameters to use for the galaxy_linking and
            classification steps?

        selector_data_filters (dict):
            Data filters to pass to select.IDSelector.select_ids()

        selector_kwargs (dict):
            Arguments to use when selecting what particles to track.
            Arguments will be passed to select.IDSelector

        sampler_kwargs (dict):
            Arguments to use when selecting what particles to track.
            Arguments will be passed to select.IDSampler

        tracker_kwargs (dict):
            Arguments to use when tracking particles.
            Arguments will be passedts to pass to track.ParticleTracker

        gal_linker_kwargs (dict):
            Arguments to use when associating particles with galaxies.
            Arguments will be passed to galaxy_link.ParticleTrackGalaxyLinker

        classifier_kwargs (dict):
            Arguments to use when classifying particles.
            Arguments will be passed to classify.Classifier

        run_id_selecting (bool):
            If True, then run routines for selecting particles.

        run_id_sampling (bool):
            If True, then run routines for sampling from the full list of
            selected particles.

        run_tracking (bool):
            If True, then run routines for tracking particles.

        run_galaxy_linking (bool):
            If True, then run routines for associating particles with galaxies.

        run_classifying (bool):
            If True, then run routines for classifying particles.
    '''

    if sim_name is not None:

        file_manager = file_management.FileManager()

        if out_dir is None:
            out_dir = file_manager.get_linefinder_dir( sim_name )

    if galdef is not None:
        galdef_dict = linefinder_config.GALAXY_DEFINITIONS[galdef]

    # Setup jugdata
    jugdir_tail = '{}.jugdata'.format( tag )
    jug.set_jugdir( os.path.join( out_dir, jugdir_tail ) )

    print( "Starting jug thread..." )

    # These are kwargs that could be used at any stage of running linefinder.
    general_kwargs = {
        'out_dir': out_dir,
        'tag': tag,
    }

    # Run the ID Selecting
    if run_id_selecting:

        # Update arguments
        selector_kwargs = utilities.merge_two_dicts(
            selector_kwargs, general_kwargs )

        if sim_name is not None:
            snapshot_kwargs = selector_kwargs['snapshot_kwargs']

            if 'sdir' not in snapshot_kwargs:
                snapshot_kwargs['sdir'] = file_manager.get_sim_dir( sim_name )

            if 'halo_data_dir' not in snapshot_kwargs:
                snapshot_kwargs['halo_data_dir'] = file_manager.get_halo_dir( sim_name )

            if 'main_halo_id' not in snapshot_kwargs:
                snapshot_kwargs['main_halo_id'] = linefinder_config.MAIN_MT_HALO_ID[sim_name]

        id_selector = select.IDSelector( **selector_kwargs )
        id_selector.select_ids_jug( selector_data_filters )

    # Run the ID Sampling
    if run_id_sampling:

        # Update arguments
        sampler_kwargs = utilities.merge_two_dicts(
            sampler_kwargs, general_kwargs )

        id_sampler = select.IDSampler( **sampler_kwargs )

        jug.Task( id_sampler.sample_ids )

        jug.barrier()

    # Run the Particle Tracking
    if run_tracking:

        # Update arguments
        tracker_kwargs = utilities.merge_two_dicts(
            tracker_kwargs, general_kwargs )

        if 'sdir' not in tracker_kwargs:
            tracker_kwargs['sdir'] = file_manager.get_sim_dir( sim_name )

        particle_tracker = track.ParticleTracker( **tracker_kwargs )
        particle_tracker.save_particle_tracks_jug()

    # Run the Galaxy Finding
    if run_galaxy_linking:

        # Update arguments
        gal_linker_kwargs = utilities.merge_two_dicts(
            gal_linker_kwargs, general_kwargs )

        if sim_name is not None:

            if 'halo_data_dir' not in gal_linker_kwargs:
                gal_linker_kwargs['halo_data_dir'] = file_manager.get_halo_dir( sim_name )

            if 'main_mt_halo_id' not in gal_linker_kwargs:
                gal_linker_kwargs['main_mt_halo_id'] = linefinder_config.MAIN_MT_HALO_ID[sim_name]

        if galdef is not None:
            for key in [ 'galaxy_cut', 'length_scale', 'mt_length_scale' ]:
                gal_linker_kwargs[key] = galdef_dict[key]

        particle_track_gal_linker = galaxy_link.ParticleTrackGalaxyLinker(
            **gal_linker_kwargs )
        particle_track_gal_linker.find_galaxies_for_particle_tracks_jug()

    # Run the Classification
    if run_classifying:

        # Update arguments
        classifier_kwargs = utilities.merge_two_dicts(
            classifier_kwargs, general_kwargs )

        if galdef is not None:
            for key in [ 't_pro', 't_m', ]:
                classifier_kwargs[key] = galdef_dict[key]

        classifier = classify.Classifier( **classifier_kwargs )
        jug.Task( classifier.classify_particles )

########################################################################
