Troubleshooting
===============

Common problems, and how to handle them.

HDF5 Import Errors
------------------

A common import error is: ::

    ImportError: libhdf5.so.10: cannot open shared object file: No such file or directory

To solve this ensure you have the right HDF5 library loaded.
For example in the case of Python3 on Stampede 2 you probably need ``module load phdf5`` instead of ``module load hdf5``.
