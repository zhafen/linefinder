#!/bin/bash

# Generate docs using the docstrings
sphinx-apidoc -o . ../linefinder -e -f

# Generate derived data quantities docs
python gen_derived_data_doc.py linefinder.analyze_data.worldlines.Worldlines ./derived_data_quantities.rst

# Make the files
make clean
make html
