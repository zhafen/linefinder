#!/usr/bin/env python
'''Main file for running pathfinder.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import select
import track
import galaxy_find
import classify

########################################################################
########################################################################


def run_pathfinder(
    selector_data_filters = None,
    selector_kwargs = None,
    sampler_kwargs = None,
    tracker_kwargs = None,
    gal_finder_kwargs = None,
    classifier_kwargs = None,
    run_id_selection = True,
    run_id_sampling = True,
    run_tracking = True,
    run_galaxy_finding = True,
    run_classifying = True,
):
    '''Main function for running pathfinder.

    Args:
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

        gal_finder_kwargs (dict):
            Arguments to use when associating particles with galaxies.
            Arguments will be passed to galaxy_find.ParticleTrackGalaxyFinder

        classifier_kwargs (dict):
            Arguments to use when classifying particles.
            Arguments will be passed to classify.Classifier

        run_id_selection (bool):
            If True, then run routines for selecting particles.

        run_id_sampling (bool):
            If True, then run routines for sampling from the full list of
            selected particles.

        run_tracking (bool):
            If True, then run routines for tracking particles.

        run_galaxy_finding (bool):
            If True, then run routines for associating particles with galaxies.

        run_classifying (bool):
            If True, then run routines for classifying particles.

    '''

    # Run the ID Selecting
    if run_id_selection:
        id_selector = select.IDSelector( **selector_kwargs )
        id_selector.select_ids( selector_data_filters )

    # Run the ID Sampling
    if run_id_sampling:
        id_sampler = select.IDSampler( **sampler_kwargs )
        id_sampler.sample_ids()

    # Run the Particle Tracking
    if run_tracking:
        particle_tracker = track.ParticleTracker( **tracker_kwargs )
        particle_tracker.save_particle_tracks()

    # Run the Galaxy Finding
    if run_galaxy_finding:
        particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder(
            **gal_finder_kwargs )
        particle_track_gal_finder.find_galaxies_for_particle_tracks()

    # Run the Classification
    if run_classifying:
        classifier = classify.Classifier( **classifier_kwargs )
        classifier.classify_particles()

########################################################################


def run_pathfinder_jug(
    selector_data_filters = None,
    selector_kwargs = None,
    sampler_kwargs = None,
    tracker_kwargs = None,
    gal_finder_kwargs = None,
    classifier_kwargs = None,
    run_id_selection = True,
    run_id_sampling = True,
    run_tracking = True,
    run_galaxy_finding = True,
    run_classifying = True,
):
    '''Main function for running pathfinder.

    Args:
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

        gal_finder_kwargs (dict):
            Arguments to use when associating particles with galaxies.
            Arguments will be passed to galaxy_find.ParticleTrackGalaxyFinder

        classifier_kwargs (dict):
            Arguments to use when classifying particles.
            Arguments will be passed to classify.Classifier

        run_id_selection (bool):
            If True, then run routines for selecting particles.

        run_id_sampling (bool):
            If True, then run routines for sampling from the full list of
            selected particles.

        run_tracking (bool):
            If True, then run routines for tracking particles.

        run_galaxy_finding (bool):
            If True, then run routines for associating particles with galaxies.

        run_classifying (bool):
            If True, then run routines for classifying particles.

    '''

    # Run the ID Selecting
    if run_id_selection:
        id_selector = select.IDSelector( **selector_kwargs )
        id_selector.select_ids( selector_data_filters )

    # Run the ID Sampling
    if run_id_sampling:
        id_sampler = select.IDSampler( **sampler_kwargs )
        id_sampler.sample_ids()

    # Run the Particle Tracking
    if run_tracking:
        particle_tracker = track.ParticleTracker( **tracker_kwargs )
        particle_tracker.save_particle_tracks()

    # Run the Galaxy Finding
    if run_galaxy_finding:
        particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder(
            **gal_finder_kwargs )
        particle_track_gal_finder.find_galaxies_for_particle_tracks()

    # Run the Classification
    if run_classifying:
        classifier = classify.Classifier( **classifier_kwargs )
        classifier.classify_particles()