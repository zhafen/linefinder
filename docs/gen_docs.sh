#!/bin/bash

# Generate docs using the docstrings
sphinx-apidoc -o . ../linefinder -e -f

# Make the files
make clean
make html
