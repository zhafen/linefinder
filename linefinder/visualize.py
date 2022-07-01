#!/usr/bin/env python
'''Tools for categorizing particles into different accretion modes.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

from .analyze_data import worldlines as analyze_worldlines
from .analyze_data import plot_worldlines

########################################################################

def export_to_firefly(
    export_to_firefly_kwargs = {},
    install_firefly = True,
    write_startup = 'append',
    include_instantaneous = True,
    include_pathlines = True,
    w = None,
    **kwargs
):
    '''Wrapper for exporting a particle tracking dataset to Firefly.

    Args:
        install_firefly (bool):
            If True, clone Firefly before exporting.

        export_to_firefly_kwargs (dict):
            Arguments to be passed to
            analyze_data.plot_worldlines.WorldlinesPlotter.export_to_firefly

        kwargs:
            Arguments to be used for loading Worldlines
    '''

    # Set up data objects
    if w is None:
        w = analyze_worldlines.Worldlines(
            **kwargs
        )
    w_plotter = plot_worldlines.WorldlinesPlotter( w )

    # Make fiducial visualization
    if include_instantaneous:
        w_plotter.export_to_firefly(
            install_firefly = install_firefly,
            write_startup = write_startup,
            **export_to_firefly_kwargs
        )

    # Make a pathlines visualization
    if include_pathlines:
        w_plotter.export_to_firefly(
            pathlines = True,
            install_firefly = install_firefly and not include_instantaneous,
            **export_to_firefly_kwargs
        )
