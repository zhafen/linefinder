#!/usr/bin/env python
'''Constants for use in displaying.
For example, this contains the colors I will want to consistently use for
different categories of accretion, as well as their names.

@author: Zach Hafen, Daniel Angles-Alcazar
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import matplotlib.colors as colors
import palettable

########################################################################
########################################################################

CLASSIFICATION_LABELS = {
    None: 'All',
    'all': 'All',
    'is_pristine': 'NEP',
    'is_NEP_NYA': 'Fresh Accretion',
    'is_fresh_accretion': 'Fresh Accretion',
    'is_NEP_wind_recycling': 'NEP Wind Recycling',
    'is_IP': 'IP',
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
    'is_hitherto_EP_NYA': 'EP CGM',
    'is_hitherto_NEP_NYA': 'NEP CGM',
    'is_in_CGM' : 'All CGM',
    'is_CGM_NEP': 'IGM Accretion',
    'is_CGM_IP': 'Wind',
    'is_CGM_EP': 'Satellite Wind',
    'is_CGM_satellite': 'Satellite ISM',
    'is_CGM_IGM_accretion': 'IGM Accretion',
    'is_CGM_wind': 'Wind',
    'is_CGM_satellite_wind': 'Satellite Wind',
    'is_CGM_satellite_ISM': 'Satellite ISM',
    'is_outside_any_gal_EP': 'Satellite Wind',
    'is_outside_any_gal_IP': 'Wind',
    'is_CGM_to_IGM': 'ejected',
    'is_CGM_to_gal_or_interface': 'accreted',
    'is_CGM_still': 'remains CGM',
    'is_CGM_accreted': 'accreted',
    'is_CGM_ejected': 'ejected',
    'is_CGM_accreted_to_satellite': 'accreted - satellite',
    'is_CGM_halo_transfer': 'halo transfer',
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
    'is_hitherto_EP_NYA': '#a65628',
    'is_hitherto_NEP_NYA': '#ffff33',
}

# This is the revised colorscheme.
# Externally-processed are warm colors, Non-externally-processed are cool colors
# Wind is on the green border of EP and NEP, and unaccreted is on the purple
# border of EP and NEP

# This set is revised again to have greater variation in brightness
# Revised to have greater variation in brightness
CLASSIFICATION_COLORS_B = {
    'all' : 'black',
    None: 'black',
    'is_in_CGM': 'black',
    'is_CGM_IGM_accretion': colors.hsv_to_rgb( np.array([ 200./360., 0.72,  0.7 ]) ),
    'is_CGM_NEP': colors.hsv_to_rgb( np.array([ 200./360., 0.72,  0.7 ]) ),
    'is_hitherto_NEP': colors.hsv_to_rgb( np.array([ 200./360., 0.72,  0.7 ]) ),
    'is_CGM_wind': colors.hsv_to_rgb( np.array([ 140./360., 0.72, 0.8 ]) ),
    'is_CGM_IP': colors.hsv_to_rgb( np.array([ 140./360., 0.72, 0.8 ]) ),
    'is_outside_any_gal_IP': colors.hsv_to_rgb( np.array([ 140./360., 0.72, 0.8 ]) ),
    'is_CGM_satellite_wind': colors.hsv_to_rgb( np.array([ 32.6/360.,  0.72,  0.7 ]) ),
    'is_CGM_EP': colors.hsv_to_rgb( np.array([ 32.6/360.,  0.72,  0.7 ]) ),
    'is_hitherto_EP': colors.hsv_to_rgb( np.array([ 32.6/360.,  0.72,  0.7 ]) ),
    'is_outside_any_gal_EP': colors.hsv_to_rgb( np.array([ 32.6/360.,  0.72,  0.7 ]) ),
    'is_CGM_satellite_ISM': colors.hsv_to_rgb( np.array([ 9.4/360,  0.8, 0.5 ]) ),
    'is_CGM_satellite': colors.hsv_to_rgb( np.array([ 9.4/360,  0.8, 0.5 ]) ),
    'is_in_galaxy_halo_interface': colors.hsv_to_rgb( np.array([ 302./360, .6, .8 ]) ),
    # These are older colors used for Galaxy Data
    'is_pristine': list( colors.hex2color( '#007AAF' ) ),
    'is_NEP_NYA': list( colors.hex2color( '#007AAF' ) ),
    'is_fresh_accretion': list( colors.hex2color( '#0A4BC6' ) ),
    'is_NEP_wind_recycling': list( colors.hex2color( '#07BC6E' ) ),
    'is_IP': list( colors.hex2color( '#4CAF00' ) ),
    'is_wind': list( colors.hex2color( '#4CAF00' ) ),
    'is_preprocessed': list( colors.hex2color( '#FB9319' ) ),
    'is_mass_transfer': list( colors.hex2color( '#FDD30D' ) ),
    'is_mass_transfer_NYA': list( colors.hex2color( '#FDD30D' ) ),
    'is_merger': list( colors.hex2color( '#E32F0E' ) ),
    'is_merger_NYA': list( colors.hex2color( '#E32F0E' ) ),
    'is_merger_star': list( colors.hex2color( '#E32F0E' ) ),
    'is_merger_gas': list( colors.hex2color( '#E67711' ) ),
    'is_CGM_accreted': palettable.cartocolors.qualitative.Safe_7.mpl_colors[0],
    'is_CGM_ejected': palettable.cartocolors.qualitative.Safe_7.mpl_colors[1],
    'is_CGM_still': palettable.cartocolors.qualitative.Safe_7.mpl_colors[2],
    'is_CGM_accreted_to_satellite': palettable.cartocolors.qualitative.Safe_7.mpl_colors[4],
    'is_CGM_halo_transfer': palettable.cartocolors.qualitative.Safe_7.mpl_colors[5],
    'is_CGM_fate_unclassified': palettable.cartocolors.qualitative.Safe_7.mpl_colors[3],
    'will_leaves_gal_dt_0.050': palettable.cartocolors.qualitative.Vivid_10.mpl_colors[0],
    'is_cluster_star': palettable.cartocolors.qualitative.Vivid_10.mpl_colors[1],
}
# This set is revised again to have greater variation in brightness
# Revised to have greater variation in brightness
CLASSIFICATION_COLORS_D = {
    None: 'black',
    'is_CGM_NEP': colors.hsv_to_rgb( np.array([ 158.4/360.,  1.0,  0.89 ]) ),
    'is_CGM_IP': colors.hsv_to_rgb( np.array([ 299./360,  0.5, 0.63 ]) ),
    'is_CGM_EP': colors.hsv_to_rgb( np.array([ 331.7/360.,  0.3,  0.45 ]) ),
    'is_CGM_satellite': colors.hsv_to_rgb( np.array([ 0.0,  1.0,  0.38 ]) ),
}

# These are the colors that were revised only once
CLASSIFICATION_COLORS_C = {
    None: 'black',
    'all': 'black',
    'is_pristine': '#007AAF',
    'is_NEP_NYA': '#007AAF',
    'is_fresh_accretion': '#0A4BC6',
    'is_NEP_wind_recycling': '#07BC6E',
    'is_IP': '#4CAF00',
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
    'is_hitherto_EP_NYA': '#FB9319',
    'is_hitherto_NEP_NYA': '#007AAF',
    'is_CGM_NEP': '#007AAF',
    'is_CGM_EP': '#FB9319',
    'is_CGM_IP': '#4CAF00',
    'is_CGM_satellite': '#E32F0E',
    'is_outside_any_gal_EP': '#FB9319',
    'is_outside_any_gal_IP': '#4CAF00',
}

# This is used in some cases, e.g. when making bar plots, to make them a little
# easier to look at.
CLASSIFICATION_ALPHA = 1.0

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

# This set of classifications is centered on the origin of the CGM
CLASSIFICATIONS_CGM_ORIGIN = [
    'is_CGM_IGM_accretion',
    'is_CGM_wind',
    'is_CGM_satellite_wind',
    'is_CGM_satellite_ISM',
]
CLASSIFICATIONS_CGM_ORIGIN_OLD = [
    'is_CGM_NEP',
    'is_CGM_IP',
    'is_CGM_EP',
    'is_CGM_satellite',
]

# This set of classifications is centered on the fate of the CGM
CLASSIFICATIONS_CGM_FATE = [
    'is_CGM_still',
    'is_CGM_accreted',
    'is_CGM_accreted_to_satellite',
    'is_CGM_ejected',
    'is_CGM_halo_transfer',
]
