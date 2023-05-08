'''Perform particle classification.'''

import sys

import linefinder.classify as classify

import trove

########################################################################

pm = trove.link_params_to_config(
    config_fp = sys.argv[1],
)

classifier = classify.Classifier(
    out_dir = pm['data_dir'],
    tag = pm['tag'],
    halo_data_dir = pm['halo_data_dir'],
)
classifier.classify_particles()
