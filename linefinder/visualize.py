#!/usr/bin/env python
'''Tools for categorizing particles into different accretion modes.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

from .analyze_data import worldlines as analyze_worldlines
import linefinder.analyze_data.plot_worldlines as plot_worldlines

########################################################################

def export_to_firefly(
    firefly_dir,
    export_to_firefly_kwargs = {},
    **kwargs
):
    '''Wrapper for exporting a particle tracking dataset to Firefly.

    Args:
        firefly_dir (str):
            Directory that should contain the Firefly visualization.

        export_to_firefly_kwargs (dict):
            Arguments to be passed to
            analyze_data.plot_worldlines.WorldlinesPlotter.export_to_firefly

        kwargs:
            Arguments to be used for loading Worldlines
    '''

    # Set up data objects
    w = analyze_worldlines.Worldlines(
        **kwargs
    )
    w_plotter = plot_worldlines.WorldlinesPlotter( w )

    # Make fiducial visualization
    w_plotter.export_to_firefly(
        firefly_dir = firefly_dir,
        install_firefly = True,
        write_startup = True,
        **export_to_firefly_kwargs
    )

    # Make a pathlines visualization
    w_plotter.export_to_firefly(
        firefly_dir = firefly_dir,
        pathlines = True,
        **export_to_firefly_kwargs
    )
