#!/usr/bin/env python
'''Constants for use in displaying.
For example, this contains the colors I will want to consistently use for
different categories of accretion, as well as their names.

@author: Zach Hafen, Daniel Angles-Alcazar
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

########################################################################
########################################################################

CLASSIFICATION_LABELS = {
    'all': 'All',
    'is_pristine': 'NEP',
    'is_NEP_NYA': 'Fresh Accretion',
    'is_fresh_accretion': 'Fresh Accretion',
    'is_NEP_wind_recycling': 'NEP Wind Recycling',
    'is_wind': 'Wind',
    'is_preprocessed': 'EP',
    'is_mass_transfer': 'Intergalactic Transfer',
    'is_mass_transfer_NYA': 'Intergalactic Transfer',
    'is_merger': 'Merger',
    'is_merger_NYA': 'Merger',
    'is_merger_star': 'Merger-Stellar',
    'is_merger_gas': 'Merger-ISM',
    'is_unaccreted': 'Unaccreted',
    'is_unaccreted_EP': 'Unaccreted-EP',
    'is_unaccreted_NEP': 'Unaccreted-NEP',
}

# These are in line with the colors used in Angles-Alcazar2017
CLASSIFICATION_COLORS_A = {
    'all': 'black',
    'is_pristine': '#984ea3',
    'is_NEP_NYA': '#984ea3',
    'is_fresh_accretion': '#984ea3',
    'is_NEP_wind_recycling': '#377eb8',
    'is_wind': '#377eb8',
    'is_preprocessed': '#4daf4a',
    'is_mass_transfer': '#4daf4a',
    'is_mass_transfer_NYA': '#4daf4a',
    'is_merger': '#e41a1c',
    'is_merger_NYA': '#e41a1c',
    'is_merger_star': '#e41a1c',
    'is_merger_gas': '#ff7f00',
    'is_unaccreted': '#a65628',
    'is_unaccreted_EP': '#a65628',
    'is_unaccreted_NEP': '#ffff33',
}

# This is the revised colorscheme.
# Externally-processed are warm colors, Non-externally-processed are cool colors
# Wind is on the green border of EP and NEP, and unaccreted is on the purple
# border of EP and NEP
CLASSIFICATION_COLORS_B = {
    'all': 'black',
    'is_pristine': '#007AAF',
    'is_NEP_NYA': '#007AAF',
    'is_fresh_accretion': '#0A4BC6',
    'is_NEP_wind_recycling': '#07BC6E',
    'is_wind': '#4CAF00',
    'is_preprocessed': '#FB9319',
    'is_mass_transfer': '#FDD30D',
    'is_mass_transfer_NYA': '#FDD30D',
    'is_merger': '#E32F0E',
    'is_merger_NYA': '#E32F0E',
    'is_merger_star': '#E32F0E',
    'is_merger_gas': '#E67711',
    'is_unaccreted': '#A400AF',
    'is_unaccreted_NEP': '#6F06C6',
    'is_unaccreted_EP': '#C60954',
}

# This is used in some cases, e.g. when making bar plots, to make them a little
# easier to look at.
CLASSIFICATION_ALPHA = 0.7

########################################################################
# Lists of Classifications Used Together
########################################################################

CLASSIFICATIONS_ALL = [
    'all',
    'is_pristine',
    'is_fresh_accretion',
    'is_NEP_wind_recycling',
    'is_wind',
    'is_preprocessed',
    'is_mass_transfer',
    'is_merger',
    'is_merger_star',
    'is_merger_gas',
    'is_unaccreted',
    'is_unaccreted_EP',
    'is_unaccreted_NEP',
]

CLASSIFICATIONS_A = [
    'is_fresh_accretion',
    'is_NEP_wind_recycling',
    'is_mass_transfer',
    'is_merger_gas',
    'is_merger_star',
]
CLASSIFICATIONS_B = [
    'is_fresh_accretion',
    'is_NEP_wind_recycling',
    'is_mass_transfer',
    'is_merger',
]

CLASSIFICATIONS_A_SMOOTH_ACCRETION = [
    'is_fresh_accretion',
    'is_NEP_wind_recycling',
    'is_mass_transfer',
]

CLASSIFICATIONS_A_GAS = [
    'is_fresh_accretion',
    'is_NEP_wind_recycling',
    'is_mass_transfer',
    'is_merger_gas',
]

# This set of classifications is centered around how gas got to the main galaxy
CLASSIFICATIONS_CGM_A = [
    'is_merger',
    'is_mass_transfer',
    'is_pristine',
    'is_unaccreted',
]

# This set of classifications is centered around how gas got to the CGM
CLASSIFICATIONS_CGM_B = [
    'is_merger_NYA',
    'is_mass_transfer_NYA',
    'is_wind',
    'is_NEP_NYA',
    'is_unaccreted_NEP',
    'is_unaccreted_EP',
]

########################################################################
# Functions for easier handling of presentation constants
########################################################################


def get_classifications_variations(
    classification_list,
    classification_colors=CLASSIFICATION_COLORS_B
):
    '''Given a list of classification, fill a dictionary with the corresponding
    colors and labels.
    '''

    classifications_variations = {}
    for key in classification_list:

        variation = {}

        # Make the classification applied
        if key != 'all':
            item['classification'] = key
            item['plot_label'] = None

        # Give it a color
        item['color'] = classification_colors[key]

        # Give it a label
        item['line_label'] = p_constants.CLASSIFICATION_LABELS[key]

        classifications_variations[classification] = variation

    return classifications_variations
