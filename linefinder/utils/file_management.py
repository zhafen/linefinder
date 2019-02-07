#!/usr/bin/env python
'''Simple functions and variables for easily accessing common files and choices
of parameters.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import os

import linefinder.config as config

########################################################################
########################################################################


class FileManager( object ):

    def __init__( self, project=None ):
        self.project = project

        parameters_name = '{}_PARAMETERS'.format( config.ACTIVE_SYSTEM.upper() )
        self.system_parameters = getattr( config, parameters_name )

        if project is not None:
            self.project_parameters = \
                self.system_parameters['project'][self.project]

    ########################################################################
    ########################################################################

    def get_sim_subdir( self, sim_name, physics=None, resolution=None, ):

        if physics is None:
            name_mapping = {
                '': 'core',
                '_md': 'metal_diffusion',
            }
            physics = name_mapping[sim_name[4:]]

        return os.path.join(
            physics,
            '{}_res{}'.format(
                sim_name[:4],
                config.DEFAULT_SIM_RESOLUTIONS[sim_name],
            )
        )

    ########################################################################

    def get_sim_dir( self, sim_name ):

        return os.path.join(
            self.system_parameters['simulation_data_dir'],
            self.get_sim_subdir( sim_name ),
            'output',
        )

    ########################################################################

    def get_metafile_dir( self, sim_name ):

        return os.path.join(
            self.system_parameters['simulation_data_dir'],
            self.get_sim_subdir( sim_name ),
        )

    ########################################################################

    def get_halo_dir( self, sim_name ):

        return os.path.join(
            self.system_parameters['halo_data_dir'],
            self.get_sim_subdir( sim_name ),
            'halo',
        )

    ########################################################################

    def get_linefinder_dir( self, sim_name, subdir='data', ):
        
        return os.path.join(
            self.system_parameters['linefinder_data_dir'],
            self.get_sim_subdir( sim_name ),
            subdir,
        )

    ########################################################################

    def get_project_figure_dir( self ):

        return os.path.join(
            self.project_parameters['project_dir'],
            'figures',
        )

    ########################################################################

    def get_project_presentation_dir( self ):

        return self.project_parameters['presentation_dir']

    ########################################################################
    ########################################################################

    def get_linefinder_analysis_defaults(
        self,
        tag_tail,
        sim_name = 'm12i',
        ahf_index = 600,
        ids_tag_tail = None,
        ptracks_tag_tail = None,
        galdef = '',
    ):
        '''Standard defaults for linefinder analysis routines.

        Args:
            tag_tail (str) :
                The second half of the tag, after the simulation name.

            sim_name (str) :
                Name of the simulation to use.

        Returns:
            defaults (dict) :
                Commonly used default dictionary.
        '''

        tag = '{}{}'.format( sim_name, tag_tail )

        if galdef != '':
            used_tag = '{}{}'.format( tag, galdef )
        else:
            used_tag = tag

        defaults = {
            'data_dir': self.get_linefinder_dir( sim_name ),
            'tag': used_tag,

            'halo_data_dir': self.get_halo_dir( sim_name ),
            'ahf_index': ahf_index,
            'main_halo_id': config.MAIN_MT_HALO_ID[sim_name],
        }

        if galdef != '':
            defaults['ids_tag'] = tag
            defaults['ptracks_tag'] = tag

        if ids_tag_tail is not None:
            defaults['ids_tag'] = '{}{}'.format( sim_name, ids_tag_tail )

        if ptracks_tag_tail is not None:
            defaults['ptracks_tag'] = '{}{}'.format(
                sim_name,
                ptracks_tag_tail,
            )

        return defaults

    ########################################################################

    def get_linefinder_analysis_variations(
        self,
        tag_tail,
        default_sim_name = 'm12i',
        sim_names = [ 'm12i', 'm12m', 'm12f', 'm12imd' ],
        *args, **kwargs
    ):
        '''Standard default variations for linefinder analysis routines.

        Args:
            tag_tail (str) :
                The second half of the tag, after the simulation name.

            default_sim_name (str) :
                Name of the simulation that's the "default".

            sim_names (list of strs) :
                What simulations to include.

            *args, **kwargs :
                Other arguments passed to
                self.get_linefinder_analysis_defaults()

        Returns:
            variations (dict of dicts) :
                Commonly used variations dictionary.
        '''

        variations = {}
        for sim_name in sim_names:

            if sim_name != default_sim_name:
                variations[sim_name] = self.get_linefinder_analysis_defaults(
                    tag_tail = tag_tail,
                    sim_name = sim_name,
                    *args, **kwargs
                )
            else:
                variations[sim_name] = {}

        return variations

    ########################################################################

    def get_linefinder_analysis_defaults_and_variations(
        self,
        tag_tail,
        default_sim_name = 'm12i',
        sim_names = [ 'm12i', 'm12m', 'm12f', ],
        *args, **kwargs
    ):
        '''Standard defaults and variations for linefinder analysis routines.

        Args:
            tag_tail (str) :
                The second half of the tag, after the simulation name.

            default_sim_name (str) :
                Name of the simulation that's the "default".

            sim_names (list of strs) :
                What simulations to include.

            *args, **kwargs :
                Other arguments passed to
                self.get_linefinder_analysis_defaults() and
                self.get_linefinder_analysis_variations() and

        Returns:
            variations (dict of dicts) :
                Commonly used variations dictionary.
        '''

        linefinder_analysis_defaults = self.get_linefinder_analysis_defaults(
            tag_tail,
            sim_name = default_sim_name,
            *args, **kwargs
        )

        linefinder_analysis_variations = \
            self.get_linefinder_analysis_variations(
                tag_tail,
                default_sim_name = default_sim_name,
                sim_names = sim_names,
                *args, **kwargs
            )

        return linefinder_analysis_defaults, linefinder_analysis_variations

